# Bridge Game Postmortem Chatbot

Project to provide high-level postmortem game information using a chat interface. User can either use vetted prompts, such as via the Summarize button, or custom prompts. Games are limited to ACBL club pair matchpoint games which used a Mitchell movement and not shuffled.

Try it live at: https://acbl.postmortem.chat/

## Related Projects and Documents
GitHub: https://github.com/BSalita/acbl-postmortem

For a list of related projects and documents see: https://github.com/BSalita/BridgeStats-ACBL

## Shareable URLs

The sidebar options are mirrored into the URL query string so any view is
shareable / bookmarkable. Supported parameters:

| Param            | Maps to sidebar option         | Example value |
|------------------|--------------------------------|---------------|
| `player_id`      | ACBL player number             | `2663279`     |
| `session_id`     | Club game or tournament session| `6534522`     |
| `show_sql_query` | Developer Settings → Show SQL  | `1` / `0`     |
| `sd_samples`     | Single Dummy Samples Count     | `10`          |

Example: `https://acbl.postmortem.chat/?player_id=2663279&session_id=6534522`
loads that exact game directly.

## Automation: email a daily PDF report

`acbl_postmortem_generator.py` drives the live app in a headless browser,
captures the generated PDF, and emails it. Combined with the included
PowerShell helper this becomes a daily "did the player play today? if so,
mail me the report" job.

One-time setup:

```powershell
pip install -r requirements.txt
playwright install chromium
```

Then put SMTP credentials in a `.env` file next to `app.py`. Gmail with an
App Password is the path that's tested and working:

```dotenv
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASSWORD=xxxxxxxxxxxxxxxx     # 16-char Google app password, no spaces
SMTP_FROM=you@gmail.com
```

Important Gmail gotchas:

- `SMTP_USER` and `SMTP_FROM` must match the Google account that owns the app
  password. You cannot use a Yahoo (or other) address as the sender on a
  Gmail-authenticated connection -- Gmail will reject the message.
- 2-Step Verification must be enabled on that Google account; otherwise the
  "App passwords" page won't be available and any password you paste will
  fail with `535-5.7.8 Username and Password not accepted`.
- The app password is **16 characters with no spaces** (Google shows it with
  spaces just for readability -- strip them when saving to `.env`).

### Send a report on demand

Email the player's most recent game right now (uses `.env`):

```powershell
python acbl_postmortem_generator.py `
    --url   "https://acbl.postmortem.chat/?player_id=2663279" `
    --email coach@example.com
```

Force a re-send even if the cache says it already went out, and print the
full SMTP conversation if something looks wrong:

```powershell
python acbl_postmortem_generator.py `
    --url   "https://acbl.postmortem.chat/?player_id=2663279" `
    --email coach@example.com `
    --force-email `
    --smtp-debug
```

Generate the PDF locally without sending email (handy for testing the
browser side in isolation):

```powershell
python acbl_postmortem_generator.py `
    --url   "https://acbl.postmortem.chat/?player_id=2663279" `
    --email me@example.com `
    --output report.pdf `
    --no-email
```

Pass SMTP settings on the command line instead of via `.env`:

```powershell
python acbl_postmortem_generator.py `
    --url        "https://acbl.postmortem.chat/?player_id=2663279" `
    --email      coach@example.com `
    --from-email you@gmail.com `
    --smtp-host  smtp.gmail.com `
    --smtp-port  587 `
    --smtp-user  you@gmail.com `
    --smtp-password "xxxxxxxxxxxxxxxx"
```

### Run it daily on Windows

Register a daily 6 PM scheduled task:

```powershell
.\schedule_daily_report.ps1 `
    -Url   "https://acbl.postmortem.chat/?player_id=2663279" `
    -Email coach@example.com `
    -Time  18:00
```

What the scheduled run does each evening:

1. Purges any cached PDFs in `report_cache\` whose mtime is older than the
   TTL (default 168 h / 1 week). The TTL is a sliding window: every cache
   hit refreshes the file's mtime, so an entry survives indefinitely as
   long as the player hasn't played a newer game.
2. Opens the URL in headless Chromium and waits for the report metadata.
3. Always loads the player's most recent game (no date filter).
4. Skips silently if `report_cache\{player_id}_{game_date}.pdf` already
   exists -- the cache assumes the report was already emailed for that
   game.
5. Otherwise downloads the PDF, stores it in the cache, and emails it.

Net effect: each game produces exactly one email, no matter how often the
scheduler runs in between, and no matter how many days pass between games.

Remove the scheduled task:

```powershell
.\schedule_daily_report.ps1 -Unregister
```

### Verify SMTP credentials in isolation

If a real run fails at the email step, run `test_smtp.py` to bisect whether
the problem is in the browser/PDF side or in SMTP itself. By default it
reads `SMTP_*` from your `.env`:

```powershell
# Just connect, EHLO/STARTTLS, and authenticate -- don't actually send:
python test_smtp.py --no-send

# Send a tiny test message to yourself:
python test_smtp.py
```

"# Bridge_Game_Postmortem_Chatbot" 
