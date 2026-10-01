# Live demo page

`index.html` is the case-study page served at [fraud-detection-alven.vercel.app](https://fraud-detection-alven.vercel.app).
Its scoring form posts to `/api/score`, which is `../api/score.py`; that function loads the current
`model/model_export.json` from this repository when it starts, so retraining and pushing updates the live
scores without a redeploy. The page text and charts change only when the page is redeployed.

To redeploy from a clean folder:

```bash
mkdir site && cp demo/index.html site/ && mkdir site/api && cp api/score.py site/api/
cd site && npx vercel link --project fraud-detection-alven && npx vercel deploy --prod
```

Charts are loaded from `figures/` on the `main` branch, so keep those paths stable.
