# Live demo page

`index.html` is the case-study page served at [fraud-detection-alven.vercel.app](https://fraud-detection-alven.vercel.app).
It follows the shared design of Alven's pages (tokens, fonts and components are documented in the personal
site's README). Its scoring form posts to `/api/score`, which is `../api/score.py`; that function loads the
current `model/model_export.json` from this repository's main branch when it starts, so retraining and
pushing a new export updates the live scores without a redeploy. The page text, table and charts change only
when the page is redeployed.

The deployment is made from the repository root. `vercel.json` rewrites `/` to this page and
`.vercelignore` uploads only `demo/index.html`, `api/score.py` and `vercel.json`, so no Python dependencies
are installed (the function is stdlib-only by design). The link to the Vercel project lives in `.vercel/`,
which is gitignored.

```bash
# from the repository root, once per machine
npx vercel link --project fraud-detection-alven --scope alven1 --yes
# every redeploy
npx vercel deploy --prod --scope alven1
```

Charts are loaded from `figures/` on the `main` branch, so keep those paths stable. The figures on the page
come from the 2 October 2026 retrain (the README's tables); update both together.
