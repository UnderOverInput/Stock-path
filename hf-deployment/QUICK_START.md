# 🚀 Update Your Hugging Face Space

## Fastest Method: Web Upload (2 minutes)

You already have a space! Update it to the new FastAPI backend:

1. **Go to Your Space**: https://huggingface.co/spaces/iofocus/stocks_price_prediction

2. **Upload Files**: Click "Files" tab → "Add file" → "Upload files"
   - Drag all 4 files from this folder:
     - ✅ README.md
     - ✅ requirements.txt
     - ✅ Dockerfile
     - ✅ app.py

3. **Commit**: Click "Commit changes to main"

4. **Wait**: Space builds in 5-10 minutes ⏰

## Your API URLs

Once deployed, your API will be live at:

- **Main UI**: `https://iofocus-stocks-price-prediction.hf.space`
- **API Endpoint**: `https://iofocus-stocks-price-prediction.hf.space/api/predict/AAPL`
- **API Docs**: `https://iofocus-stocks-price-prediction.hf.space/docs`
- **Health Check**: `https://iofocus-stocks-price-prediction.hf.space/api/health`

## Test Your API

```bash
# Test prediction
curl https://iofocus-stocks-price-prediction.hf.space/api/predict/AAPL

# Expected response:
{
  "ticker": "AAPL",
  "open": 185.42,
  "high": 187.89,
  "low": 184.15,
  "close": 186.73,
  "volume": 52340000,
  "rmse": 0.0423,
  "confidence": "High"
}
```

## Use in Your App (frameworx.site)

```javascript
// Frontend fetch
const response = await fetch('https://iofocus-stocks-price-prediction.hf.space/api/predict/AAPL');
const prediction = await response.json();
console.log(`Predicted close: $${prediction.close}`);
```

---

**That's it!** No code changes needed - just upload and go! 🎉
