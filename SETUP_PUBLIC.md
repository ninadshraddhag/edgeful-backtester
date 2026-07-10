# Going public — owner setup checklist (~15 minutes, one-time)

The code for Google-only login, credits and the admin panel is already deployed.
It activates itself the moment the two secrets blocks below exist. Until then the
app runs in **DEV MODE** (no login screen — safe for local use, but do the steps
below before sharing the public link).

Only YOU can do these two steps, because they create credentials on your accounts.
**Never paste these secrets into chat, email or git — only into the Streamlit
secrets screen (step 3) or your local gitignored `.streamlit/secrets.toml`.**

---

## Step 1 — Google sign-in (Google Cloud Console, ~7 min)

1. Open https://console.cloud.google.com → sign in with **ninadshraddhag@gmail.com**.
2. Top bar → project dropdown → **New project** → name `sharp-backtester` → Create → select it.
3. Left menu → **APIs & Services → OAuth consent screen**:
   - User type: **External** → Create.
   - App name `Sharp Backtester`, support email = your gmail, developer email = your gmail → Save through the steps (scopes: keep defaults; no need to add any).
   - Under **Audience / Publishing status** click **Publish app** (so ANY Gmail user can sign in, not just test users).
4. **APIs & Services → Credentials → + Create credentials → OAuth client ID**:
   - Application type: **Web application**, name `sharp-web`.
   - Authorized redirect URIs → **Add URI**, paste EXACTLY:
     `https://backtesterpro.streamlit.app/oauth2callback`
   - Create → a dialog shows **Client ID** (ends `.apps.googleusercontent.com`) and **Client secret** (starts `GOCSPX-`). Keep this tab open for step 3.

## Step 2 — Supabase credit database (~5 min)

1. Open https://supabase.com → **Start your project** → sign in (GitHub or Google) → **New project**:
   - Name `sharp-credits`, region **Mumbai (ap-south-1)**, generate a DB password (you won't need it again) → Create.
2. When it finishes provisioning: left menu → **SQL Editor → New query** → paste ALL of the SQL below → **Run**:

```sql
create table if not exists users (
  email          text primary key,
  credits        integer not null default 20,
  total_used     integer not null default 0,
  is_admin       boolean not null default false,
  accepted_terms timestamptz,
  seen_tour      boolean not null default false,
  first_seen     timestamptz not null default now(),
  last_seen      timestamptz not null default now()
);

create table if not exists usage_log (
  id      bigserial primary key,
  email   text not null,
  action  text not null,
  credits integer not null,
  ts      timestamptz not null default now()
);

alter table users enable row level security;
alter table usage_log enable row level security;
-- no public policies: only the service_role key (kept in Streamlit secrets) has access

create or replace function charge_credits(p_email text, p_n int, p_action text)
returns int language plpgsql security definer as $$
declare new_balance int;
begin
  update users set credits = credits - p_n,
                   total_used = total_used + p_n,
                   last_seen = now()
   where email = p_email and credits >= p_n
   returning credits into new_balance;
  if new_balance is null then return -1; end if;
  insert into usage_log(email, action, credits) values (p_email, p_action, p_n);
  return new_balance;
end $$;

create or replace function add_credits(p_email text, p_n int)
returns int language plpgsql security definer as $$
declare new_balance int;
begin
  update users set credits = credits + p_n where email = p_email
  returning credits into new_balance;
  insert into usage_log(email, action, credits) values (p_email, 'admin_grant', -p_n);
  return coalesce(new_balance, -1);
end $$;

insert into users(email, credits, is_admin)
values ('ninadshraddhag@gmail.com', 999999, true)
on conflict (email) do update set is_admin = true;
```

3. Left menu → **Project Settings → API**: copy the **Project URL**
   (`https://xxxx.supabase.co`) and the **service_role** key (the SECRET one,
   not `anon`). Keep the tab open for step 3.

## Step 3 — paste secrets into Streamlit Cloud (~3 min)

1. Open https://share.streamlit.io → your app → **⋮ → Settings → Secrets**.
2. Paste the block below, filling the five `<...>` values from steps 1–2.
   For `cookie_secret`, type any long random string (40+ characters — mash the
   keyboard; it just signs the login cookie).

```toml
[auth]
redirect_uri = "https://backtesterpro.streamlit.app/oauth2callback"
cookie_secret = "<LONG RANDOM STRING>"
client_id = "<CLIENT ID from step 1>"
client_secret = "<CLIENT SECRET from step 1>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

[supabase]
url = "<PROJECT URL from step 2>"
service_key = "<SERVICE_ROLE KEY from step 2>"
```

3. Save → **⋮ → Reboot app**. Done: visiting the app now shows the Google
   sign-in page; you sign in and see the **👑 Admin** mode.

---

## Daily operation

- **Grant credits**: app → 👑 Admin → pick user → amount → Grant. New signups
  automatically get 20 credits (change `STARTER_CREDITS` in `credits.py`).
- **See who signed up / usage**: same panel; CSV export buttons included.
- **Costs**: everything above is on free tiers (Google OAuth: free ·
  Supabase free tier: 500 MB, plenty · Streamlit Community Cloud: free).

## When the app gets slow / popular — upgrade path

Streamlit Community Cloud (free) = 1 container, ~1 GB RAM. Fine for a beta with
tens of users; NOT for hundreds at once.

| Stage | Users/day | Action | Cost |
|---|---|---|---|
| Beta (now) | < 30 | stay on Community Cloud | ₹0 |
| Growth | 30–150 | move to Render.com Standard or a Hetzner VPS (4 GB), same repo, `streamlit run app.py` | ~$19/mo or ~€6/mo |
| Scale | 150+ | bigger VPS + split heavy backtests into a worker/queue | ~$40+/mo, needs a dev pass |

Trigger to upgrade: pages feel sluggish at Indian market open, or the app sleeps
and cold-starts often. The move is copy-paste (repo + secrets), zero code change.
