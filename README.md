# SimuScale — Replit setup

Follow these steps **exactly** and it will work. If anything goes wrong, the troubleshooting section at the bottom covers every common issue.

---

## Step 1 — Get your NVIDIA API key (skip if you already have one)

1. Go to **https://build.nvidia.com**
2. Sign up (free)
3. Click any Llama model
4. Click **"Get API Key"** in the top right
5. Copy the key — it starts with `nvapi-`

Keep this tab open. You'll paste the key in Step 4.

---

## Step 2 — Create the Repl

1. Go to **https://replit.com**
2. Click **"+ Create Repl"**
3. Pick the **Python** template
4. Name it `simuscale` (or anything)
5. Click **"Create Repl"**

---

## Step 3 — Upload the files

In the Replit file panel on the left:

1. **Delete** the default `main.py` that Replit created (right-click → Delete)
2. **Drag and drop these 4 files** from your computer into the file panel:
   - `main.py`
   - `requirements.txt`
   - `.replit`
   - `README.md` (this file — optional)

> ⚠️ **The `.replit` file is the most important one.** It's a hidden file. If you don't see it after dragging, click the three-dot menu in the file panel → **"Show hidden files"**. Without this file, Replit will try to run Python normally and fail.

---

## Step 4 — Add your API key as a secret

1. In the left sidebar, click **🔒 Secrets** (lock icon — may be under "Tools")
2. Click **"+ New Secret"**
3. **Key:** `NVIDIA_API_KEY` (exactly this, all caps, no spaces)
4. **Value:** paste your `nvapi-...` key
5. Click **"Add Secret"**

---

## Step 5 — Run it

1. Click the green **▶ Run** button at the top
2. Wait ~30 seconds (first run installs Streamlit + the OpenAI library — slow only the first time)
3. A webview pane will open on the right with your app

That's it. You're running.

---

## Troubleshooting

### "It says 'streamlit: command not found'"
Dependencies didn't install. In the **Shell** tab at the bottom, run:
```
pip install -r requirements.txt
```
Then click Run again.

### "The webview is blank / says connection refused"
The `.replit` file isn't there or has the wrong content. Make sure hidden files are visible (three-dot menu → Show hidden files), then check that `.replit` exists and matches the file I gave you.

### "Missing API key" error in the app
Your secret didn't load. Stop the Repl, go to **Secrets**, verify the key name is exactly `NVIDIA_API_KEY` (case-sensitive), then Run again.

### "404 model not found" when running a simulation
NVIDIA may have changed the model name. In the **Shell** tab, paste this and tell me what the first 20 models are:
```
python -c "from openai import OpenAI; import os; c=OpenAI(api_key=os.environ['NVIDIA_API_KEY'], base_url='https://integrate.api.nvidia.com/v1'); print([m.id for m in c.models.list().data][:20])"
```
Then I'll tell you which line of `main.py` to update.

### "The Run button does nothing"
The `.replit` file is malformed. Replace its entire contents with what I provided in the bundle — don't edit it manually.

### "It's working but I want to make the URL public"
At the top of the Replit window, click the URL bar — there's a "🔗" or "Open in new tab" button. That URL is publicly accessible to anyone with the link (no Replit login needed) for as long as the Repl is running.
