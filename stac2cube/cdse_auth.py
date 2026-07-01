"""Copernicus Data Space Ecosystem (CDSE) S3 authentication.

CDSE serves Sentinel-2 pixels from ``s3://eodata`` and requires personal S3
keys to read them (browsing/searching the STAC catalogue is anonymous; only the
actual pixel read needs the keys).

This module reads the user's keys from ``credentials/cdse_key`` (a plain
``key = value`` text file at the repository root) and exports the GDAL/AWS
environment variables that ``odc.stac`` / GDAL's ``/vsis3/`` driver use when the
data cube is finally computed.

See ``credentials/README.md`` for the (no-IT-knowledge) instructions on how to
obtain the keys.
"""

import os
from pathlib import Path

# Default CDSE S3 endpoint. The asset hrefs are of the form
# ``s3://eodata/Sentinel-2/...`` and are served from this host.
_CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"

# Where to look for the credentials file. Order:
#   1) the STAC2CUBE_CDSE_KEYFILE environment variable, if set
#   2) <repo root>/credentials/cdse_key  (repo root = parent of this package)
_DEFAULT_KEYFILE = Path(__file__).resolve().parents[1] / "credentials" / "cdse_key"

# Accepted left-hand-side names in the credentials file, mapped to a canonical
# field. Lenient on purpose so users don't get tripped up by small naming
# differences.
_ACCESS_ALIASES = {
    "access_key",
    "access_key_id",
    "aws_access_key_id",
    "accesskey",
    "key",
}
_SECRET_ALIASES = {
    "secret_key",
    "secret_access_key",
    "aws_secret_access_key",
    "secretkey",
    "secret",
}
_ENDPOINT_ALIASES = {"endpoint", "s3_endpoint", "aws_s3_endpoint", "host"}
_REGION_ALIASES = {"region", "aws_region", "aws_default_region"}


def _looks_like_placeholder(value: str) -> bool:
    """True if the value is empty or still the template placeholder text."""
    if not value:
        return True
    low = value.strip().lower()
    return ("paste" in low) or ("your-" in low) or ("your_" in low) or (
        low in {"...", "xxx", "changeme"}
    )


def _parse_keyfile(path: Path) -> dict:
    """Parse a ``key = value`` credentials file into a dict of fields."""
    fields = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().lower()
        value = value.strip().strip('"').strip("'")
        if name in _ACCESS_ALIASES:
            fields["access"] = value
        elif name in _SECRET_ALIASES:
            fields["secret"] = value
        elif name in _ENDPOINT_ALIASES:
            fields["endpoint"] = value
        elif name in _REGION_ALIASES:
            fields["region"] = value
    return fields


def read_cdse_credentials(keyfile=None):
    """Return ``(access_key, secret_key, endpoint, region)`` for CDSE.

    Resolution order:
      1) If both ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` are already
         set in the environment, use those (lets power users override the file).
      2) Otherwise read the credentials file (``keyfile`` arg, the
         ``STAC2CUBE_CDSE_KEYFILE`` env var, or the default repo location).

    Raises a clear, actionable error if no usable keys are found.

    Resolution order is deliberately FILE FIRST: a filled ``credentials/cdse_key``
    always wins over ambient ``AWS_*`` environment variables. Otherwise unrelated
    AWS credentials already present in the session (e.g. real AWS keys, or the
    placeholder values from an early manual test) silently override the CDSE keys
    and the read fails with ``InvalidAccessKeyId``. Power users with no keyfile
    can still rely on the ``AWS_*`` env fallback below.
    """
    if keyfile is None:
        keyfile = os.environ.get("STAC2CUBE_CDSE_KEYFILE", _DEFAULT_KEYFILE)
    keyfile = Path(keyfile)

    # 1) Credentials file (preferred).
    if keyfile.exists():
        fields = _parse_keyfile(keyfile)
        access = fields.get("access", "")
        secret = fields.get("secret", "")
        if not _looks_like_placeholder(access) and not _looks_like_placeholder(secret):
            endpoint = fields.get("endpoint") or _CDSE_S3_ENDPOINT
            region = fields.get("region") or "default"
            return access, secret, endpoint, region

    # 2) Ambient AWS_* env vars (fallback for users without a keyfile).
    env_access = os.environ.get("AWS_ACCESS_KEY_ID")
    env_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if env_access and env_secret and not _looks_like_placeholder(env_access):
        endpoint = os.environ.get("AWS_S3_ENDPOINT", _CDSE_S3_ENDPOINT)
        region = os.environ.get("AWS_REGION") or os.environ.get(
            "AWS_DEFAULT_REGION", "default"
        )
        return env_access, env_secret, endpoint, region

    # 3) Nothing usable -> actionable error.
    if not keyfile.exists():
        raise FileNotFoundError(
            "CDSE credentials file not found.\n"
            f"  Expected at: {keyfile}\n"
            "  Create it (see credentials/README.md) and paste your CDSE S3\n"
            "  'access_key' and 'secret_key' into it."
        )
    raise ValueError(
        "CDSE credentials are missing or still set to the placeholder text.\n"
        f"  File: {keyfile}\n"
        "  Open it and replace the placeholders with your real keys:\n"
        "      access_key = <your access key>\n"
        "      secret_key = <your secret key>\n"
        "  Don't have keys yet? See credentials/README.md for how to get\n"
        "  them for free (takes a couple of minutes)."
    )


# AWS/GDAL S3 environment variables that configure_cdse_environment() sets for
# CDSE's private s3://eodata endpoint. They MUST be cleared before reading any
# AWS-hosted public bucket (element84's sentinel-cogs L2A and the requester-pays
# sentinel-s2-l1c L1C), otherwise GDAL routes those reads to CDSE's endpoint and
# every band fails with InvalidAccessKeyId - which, under odc's fail_on_error=
# False, is silently filled with nodata (an all-zero scene). For cloud masking
# that means s2cloudless sees a blank cube and reports 0% cloud everywhere.
_CDSE_S3_ENV_VARS = (
    "AWS_S3_ENDPOINT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_HTTPS",
    "AWS_VIRTUAL_HOSTING",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
)


def configure_anonymous_aws_environment(requester_pays: bool = False):
    """Reset the GDAL/AWS S3 environment for anonymous access to AWS-hosted public
    Sentinel-2 buckets (element84).

    Symmetric counterpart to :func:`configure_cdse_environment`: it clears the
    CDSE-specific S3 settings a previous CDSE query may have left in ``os.environ``
    (endpoint, keys, path-style hosting) so the reads go to real AWS S3, then sets
    unsigned access. Idempotent; safe to call before every element84 search/load.

    ``requester_pays`` -> also set ``AWS_REQUEST_PAYER=requester`` for the L1C
    bucket (``s3://sentinel-s2-l1c``); the L2A COG bucket (``s3://sentinel-cogs``)
    is free, so it stays unset there.
    """
    for var in _CDSE_S3_ENV_VARS:
        os.environ.pop(var, None)
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
    if requester_pays:
        os.environ["AWS_REQUEST_PAYER"] = "requester"
    else:
        os.environ.pop("AWS_REQUEST_PAYER", None)


def configure_cdse_environment(keyfile=None):
    """Read the CDSE keys and export the GDAL/AWS env vars used at compute time.

    Idempotent: safe to call before every CDSE search/load. Returns the endpoint
    that was configured.
    """
    access, secret, endpoint, region = read_cdse_credentials(keyfile)

    os.environ["GDAL_HTTP_TCP_KEEPALIVE"] = "YES"
    os.environ["AWS_S3_ENDPOINT"] = endpoint
    os.environ["AWS_ACCESS_KEY_ID"] = access
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret
    os.environ["AWS_HTTPS"] = "YES"
    os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
    os.environ["AWS_DEFAULT_REGION"] = region
    os.environ["AWS_REGION"] = region

    # CDSE requires *signed* requests. Make sure no leftover settings from a
    # previous anonymous/requester-pays source (e.g. element84 L1C) interfere.
    os.environ["AWS_NO_SIGN_REQUEST"] = "NO"
    os.environ.pop("AWS_REQUEST_PAYER", None)

    # CDSE's S3 throttles / drops connections far more than AWS, especially
    # under odc+dask parallel reads. Without retries a transient failure makes
    # GDAL return a nodata-filled (all-zero) scene *silently*, so whole dates
    # come back empty at random. Retry hard and trim avoidable requests.
    # setdefault: respect anything the user has already tuned.
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "10")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
    os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "30")
    # Don't list the (huge) .SAFE directory on open - we open exact hrefs.
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    # HTTP/2 multiplexing against CDSE is a known source of corrupt/empty reads.
    os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "NO")
    os.environ.setdefault("VSI_CACHE", "TRUE")
    # Benchmarked against CDSE, the per-scene cost is NOT CPU-bound wavelet decode
    # but NETWORK round-trip latency: each .jp2 open costs several high-latency
    # requests to eodata S3, and that dominates (e.g. ~5 s/scene even for a tiny
    # AOI). So keep GDAL single-threaded per read -- extra decode threads only
    # MULTIPLY concurrent requests and trip CDSE's rate limit (HTTP 429). With
    # threads=1, total in-flight requests == the caller's dask worker count (not
    # workers x cores), which stays 429-free at ANY machine size. This is the
    # single most important CDSE knob -- do NOT raise it.
    os.environ.setdefault("GDAL_NUM_THREADS", "1")
    # Latency reducers: pull the JP2 codestream header in one shot instead of
    # several tiny range reads, and coalesce adjacent ranges -- fewer (expensive)
    # round-trips per file open.
    os.environ.setdefault("GDAL_INGESTED_BYTES_AT_OPEN", "65536")
    os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")

    return endpoint
