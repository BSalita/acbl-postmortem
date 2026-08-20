"""
One-time Cloudflare challenge solver for ACBL club results.

my.acbl.org is protected by an interactive Cloudflare Turnstile challenge
("Verify you are human") that automated browsers cannot click through. This
script opens a real Chrome window ON-SCREEN using the same persistent profile
that mlBridgeAcblLib uses for scraping. Click the checkbox when it appears;
the resulting cf_clearance cookie (observed lifetime: 1 year) is stored in the
profile, after which all automated fetches pass without any challenge.

Re-run this script only if club-results fetching starts failing again with a
"Cloudflare challenge detected" error (cookie expired or was invalidated).

Usage:
    python acbl_solve_challenge.py                  # default profile location
    set ACBL_BROWSER_PROFILE_DIR=D:\\some\\dir && python acbl_solve_challenge.py
"""
import os
import pathlib
import sys
import time

from playwright.sync_api import sync_playwright

# Deliberately self-contained: importing mlBridge.mlBridgeAcblLib would drag in
# the whole app dependency chain (logging_config, pandas, ...), which breaks
# outside the Streamlit app's sys.path setup. Keep these constants in sync with
# mlBridge/mlBridgeAcblLib.py.
ACBL_PROFILE_DIR_ENV = 'ACBL_BROWSER_PROFILE_DIR'
ACBL_DEFAULT_PROFILE_DIRS = (
    pathlib.Path('e:/bridge/data/acbl/playwright_profile'),
    pathlib.Path('playwright_profile'),
)


def resolve_acbl_browser_profile_dir():
    env_dir = os.getenv(ACBL_PROFILE_DIR_ENV)
    if env_dir:
        return pathlib.Path(env_dir)
    for candidate in ACBL_DEFAULT_PROFILE_DIRS:
        if candidate.exists():
            return candidate
    return None


CHECK_URL = "https://my.acbl.org/club-results"
SOLVE_TIMEOUT_SECONDS = 300
# Markers that appear on the Cloudflare interstitial itself — not on the live
# results page (which may still embed turnstile scripts after clearance).
INTERSTITIAL_MARKERS = (
    'just a moment',
    'checking your browser',
    'challenge-platform',
    '_cf_chl_opt',
    'cf-challenge',
    'cf_chl_',
)


def _has_cf_clearance(context) -> bool:
    return any(c['name'] == 'cf_clearance' for c in context.cookies('https://my.acbl.org'))


def is_challenged(page, context=None) -> bool:
    """Return True while the Cloudflare interstitial is still blocking access.

    Judge by the PAGE, never by cookie presence: a stale cf_clearance (e.g.
    bound to an older Chrome UA after an image rebuild) sits in the profile
    while the challenge is actively showing. Short-circuiting on the cookie
    made this script report "already cleared" and skip the interactive solve
    exactly when it was needed.
    """
    try:
        url = (page.url or '').lower()
        title = (page.title() or '').lower()
    except Exception:
        return True  # mid-navigation

    if 'just a moment' in title or 'checking your browser' in title:
        return True

    # Landed on club-results without the interstitial title → cleared.
    if 'my.acbl.org/club-results' in url:
        return False

    try:
        content = page.content().lower()
    except Exception:
        return True

    return any(m in content for m in INTERSTITIAL_MARKERS)


def main() -> int:
    profile_dir = resolve_acbl_browser_profile_dir() or ACBL_DEFAULT_PROFILE_DIRS[0]
    profile_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using Chrome profile: {profile_dir.resolve()}")

    with sync_playwright() as p:
        # Keep flags identical to mlBridgeAcblLib.create_acbl_browser_context
        # (except window position: the solve window must be on-screen so a
        # human can click). cf_clearance is bound to the browser fingerprint,
        # so solving with a different launch than the scraper's can yield a
        # cookie the scraper cannot use.
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            channel='chrome',
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-first-run',
                '--no-default-browser-check',
                '--window-size=1920,1080',
                # Chrome refuses to run as root (typical in containers) with
                # its sandbox enabled.
                *(['--no-sandbox'] if os.name == 'posix' and os.geteuid() == 0 else []),
            ],
            no_viewport=True,
        )
        page = context.pages[0] if context.pages else context.new_page()
        try:
            page.goto(CHECK_URL, wait_until='domcontentloaded', timeout=60000)

            if not is_challenged(page, context):
                print("No challenge shown - the profile is already cleared. Nothing to do.")
                return 0

            print()
            print("A Chrome window is open showing the Cloudflare challenge.")
            print('Please click the "Verify you are human" checkbox.')
            print(f"Waiting up to {SOLVE_TIMEOUT_SECONDS // 60} minutes...")
            print()

            deadline = time.time() + SOLVE_TIMEOUT_SECONDS
            last_heartbeat = time.time()
            while time.time() < deadline:
                if not is_challenged(page, context):
                    print("Challenge cleared - clearance cookie saved to the profile.")
                    for c in context.cookies('https://my.acbl.org'):
                        if c['name'] == 'cf_clearance':
                            exp = c.get('expires', -1)
                            if exp and exp > 0:
                                print(f"cf_clearance expires: {time.strftime('%Y-%m-%d', time.localtime(exp))}")
                    print("Automated club-results fetching should now work until the cookie expires.")
                    return 0
                if time.time() - last_heartbeat >= 15:
                    remaining = int(deadline - time.time())
                    cookie = 'yes' if _has_cf_clearance(context) else 'no'
                    print(
                        f"Still waiting ({remaining}s left). "
                        f"url={page.url!r} title={page.title()!r} cf_clearance={cookie}",
                        flush=True,
                    )
                    last_heartbeat = time.time()
                time.sleep(2)

            print("Timed out waiting for the challenge to be solved. Run the script again.")
            return 1
        finally:
            context.close()


if __name__ == '__main__':
    sys.exit(main())
