# Migrating Sharp to Render.com (~10 minutes)

Use this if Streamlit Community Cloud misbehaves (e.g. the public-access bug)
or when traffic outgrows the free tier. The repo already contains
`render.yaml`, so Render auto-configures everything.

## Steps (owner only — needs your accounts)

1. https://render.com → **Get Started** → **Sign in with GitHub** → authorize.
2. Dashboard → **New + → Blueprint** → select repo
   `ninadshraddhag/edgeful-backtester` → Render reads `render.yaml` → **Apply**.
3. While it builds: open the new service → **Environment → Secret Files →
   Add Secret File**:
   - Filename: `.streamlit/secrets.toml`
   - Contents: the SAME TOML block used on Streamlit Cloud (auth + supabase),
     with ONE change — set the redirect line to the Render URL:
     `redirect_uri = "https://sharp-backtester.onrender.com/oauth2callback"`
4. Google Cloud console → Auth Platform → Clients → sharp →
   **Authorised redirect URIs → + Add URI**:
   `https://sharp-backtester.onrender.com/oauth2callback`  → Save.
   (Keep the old streamlit.app URI too — both can coexist.)
5. Wait for the deploy to finish → app is live at
   `https://sharp-backtester.onrender.com` — public, no platform login wall,
   Google sign-in + credits work identically.

## Notes
- Free tier sleeps after ~15 min idle (cold start ~1 min). Starter $7/mo = always on.
- Custom domain (e.g. app.yourdomain.com) attachable later in Render settings;
  add its /oauth2callback to Google URIs when you do.
- facts.csv is committed, so first boot is instant — no rebuild needed.
