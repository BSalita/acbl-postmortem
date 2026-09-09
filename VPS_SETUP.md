# VPS setup: ACBL club results behind Cloudflare

Since July 2026, `my.acbl.org` shows an interactive Cloudflare Turnstile
challenge ("Verify you are human") that automated browsers cannot click
through. The workaround, already implemented in `mlBridge/mlBridgeAcblLib.py`,
is:

- fetch club results with a **real, headed Google Chrome** using a
  **persistent profile**;
- a human solves the checkbox **once** (the `cf_clearance` cookie lasts about
  a year, but is bound to this server's IP — it cannot be copied from another
  machine);
- all later fetches pass automatically until the cookie expires.

Tournament results are unaffected (they use the official `api.acbl.org` API).

## One-time setup on the VPS

From the app directory (`/app/postmortem-acbl`):

```bash
git pull
bash vps_solve_challenge.sh
```

The script installs Google Chrome, Xvfb, and x11vnc if missing, starts a
virtual display with a VNC server on `localhost:5900`, and opens the challenge
page in Chrome.

From your desktop, tunnel the VNC port and click the checkbox:

```bash
ssh -L 5900:localhost:5900 <user>@<vps>
```

Connect any VNC viewer (TightVNC, RealVNC) to `localhost:5900`, click
**"Verify you are human"**, and wait for the script to print the cookie
expiry date.

If the app runs inside Docker, run the script inside the container
(`docker exec -it <container> bash`), and make sure the profile directory
(default `/app/data/playwright_profile`, override with
`ACBL_BROWSER_PROFILE_DIR`) is on a mounted volume so it survives rebuilds.

## Run the app with a display

Headed Chrome needs a `DISPLAY` at fetch time. Without one, the library
silently falls back to headless Chromium, which Cloudflare blocks. Start the
app through the wrapper:

```bash
bash run_app_xvfb.sh
```

or equivalently keep an Xvfb running and set `DISPLAY` and
`ACBL_BROWSER_PROFILE_DIR` in the app's environment.

## Verifying it works

Look for this log line when a club-results fetch runs:

```
Using persistent Chrome profile: /app/data/playwright_profile
```

- Line present + fetch succeeds: done.
- Line absent: the profile dir wasn't found — check `ACBL_BROWSER_PROFILE_DIR`.
- Line present but fetch still fails with "Cloudflare challenge detected":
  Chrome couldn't launch headed (missing DISPLAY) or the cookie
  expired/invalidated — re-run `bash vps_solve_challenge.sh`.

## When it breaks again

If club-results fetching starts failing with the Cloudflare error after
working (cookie expired or invalidated), just re-run the one-time setup —
it takes about two minutes.

The durable fix is official API access for club results; see the request
email drafted for ACBL (tournament API contact: Mitch.Hodus@acbl.org).
