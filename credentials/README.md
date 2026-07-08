# How to get your FREE Copernicus (CDSE) download keys (in 5 Minutes!)

`stac2cube` can download Sentinel-2 data directly from the **Copernicus Data
Space Ecosystem (CDSE)** when you set `source="cdse"`.

To download the actual image pixels, CDSE needs two personal keys:

- an **access key** (think of it as a username)
- a **secret key** (think of it as a password)

Both are **free**. Follow the steps below once, paste the two keys into the
file `cdse_key` (in this same folder), and you are done.

---

## Step 1 - Create a free Copernicus account

1. Go to **https://dataspace.copernicus.eu/**
2. Click **Register** (top right) and create an account.
3. Confirm your email address (check your inbox for a confirmation link).

That's it - the account is free and gives you access to all the data.

---

## Step 2 - Open the key manager

1. Log in with the account you just created.
2. Go to this page:
   **https://eodata-s3keysmanager.dataspace.copernicus.eu/panel/s3-credentials**

   (You can also reach it from the dashboard: look for **"S3 credentials"**.)

---

## Step 3 - Create your keys

1. Click **Add Credentials** (or **Generate**).
2. If you are asked for an expiration date, choose something comfortable,
   for example **one year** from today.
3. Confirm.

The website will now show you an **access key** and a **secret key**.

> ⚠️ **Do NOT close this window yet!**
> The **secret key is shown only once**. If you close the window before
> copying it, you cannot see it again - you would have to delete the keys
> and create new ones.

---

## Step 4 - Paste the keys into the file

1. Open the file **`cdse_key`** that sits next to this README
   (open it with any text editor, e.g. Notepad).
2. Replace the placeholder text after each `=` sign with your real keys:

   ```
   access_key = your-access-key-here
   secret_key = your-secret-key-here
   ```

3. **Save** the file.


---



THAT IS IT! <br><br>
`stac2cube` reads the keys from `cdse_key` automatically. You do not need to
type them in your code.

---

**OPTIONALLY:** <br>Protecting this configuration from feature repository pulls is recommended, otherwise the changes will be lost!
```bash
git update-index --skip-worktree credentials/cdse_key
```

## Frequently asked

**Do I have to do this every time?**
No. You set the keys once. They keep working until the expiration date you
chose in Step 3.

**My download fails with a "403" or "Access Denied" error.**
Almost always this means the keys are missing, mistyped, or expired:
- Make sure you saved `cdse_key` after pasting.
- Check there are no extra spaces or missing characters in the keys.
- If the keys are old, create fresh ones (Step 3) and paste them again.

**Is my secret key safe?**
Treat it like a password. Don't share the `cdse_key` file and don't upload it
with your real keys to a public place (e.g. GitHub). If you ever leak it, just
delete the keys in the key manager and create new ones.
