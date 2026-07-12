# Professional deployment — AWS Lightsail (Mumbai) + your own domain

Total cost: ~$10/mo (Lightsail 2 GB) + ~₹700–1,200/yr (domain).
Owner does Phase A (accounts/payments, ~25 min). The server engineering
(Phase B) is done over SSH — hand over the key and IP, nothing else.

---

## Phase A — owner steps

### A1. Lightsail instance (~10 min)
1. https://aws.amazon.com → Create account (card required; won't be charged
   beyond the plan) — or sign into an existing account.
2. Go to https://lightsail.aws.amazon.com → **Create instance**:
   - Region: **Mumbai (ap-south-1)**
   - Platform: **Linux/Unix** → Blueprint: **OS only → Ubuntu 24.04 LTS**
   - SSH key pair: **Create new** → name `sharp-key` → **Download** the
     `.pem` file → save it to `C:\Users\ninad\sharp-key.pem`
   - Plan: **$10/mo (2 GB RAM, 2 vCPU, 60 GB SSD)** — often first 3 months free
   - Name: `sharp-backtester` → **Create instance**
3. Once running: instance page → **Networking** tab:
   - **Create static IP** → attach to the instance → note the IP
   - Firewall: **+ Add rule** → HTTPS (443). (SSH 22 and HTTP 80 exist by default.)

### A2. Domain (~10 min)
1. Buy at Namecheap / GoDaddy / Hostinger — e.g. `sharpbacktester.com`
   (or `.in` — cheaper, local flavor).
2. In the registrar's **DNS settings**, add two records (TTL: automatic):
   - `A` record · Host `@`   · Value = the static IP from A1
   - `A` record · Host `www` · Value = the same IP

### A3. Google sign-in for the new domain (~2 min)
1. https://console.cloud.google.com/auth/clients?project=sharp-backtester
   → click **sharp** → **Authorised redirect URIs → + Add URI**:
   `https://YOURDOMAIN.com/oauth2callback`  → **Save**.
   (Keep the streamlit.app URI too — both apps can run in parallel.)

### A4. Hand over
Provide: the static IP, the domain name, and the path to `sharp-key.pem`.

---

## Phase B — server setup over SSH (scripted)

```bash
# connect
ssh -i sharp-key.pem ubuntu@STATIC_IP

# 1. Docker
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker ubuntu && newgrp docker

# 2. Swap (safety net on 2 GB)
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 3. App
git clone https://github.com/ninadshraddhag/edgeful-backtester.git app && cd app
nano Caddyfile          # replace sharpbacktester.com with the real domain
nano secrets.toml       # paste the secrets TOML; set redirect_uri to the new domain

# 4. Launch (Caddy fetches TLS automatically once DNS resolves)
docker compose up -d --build

# 5. Updates later
git pull && docker compose up -d --build
```

## After go-live
- Streamlit Cloud app can stay running as a free backup/staging URL.
- UptimeRobot (free) → monitor https://YOURDOMAIN.com, email on downtime.
- Upgrade path: Lightsail resize to 4 GB ($20) is two clicks + reboot.
