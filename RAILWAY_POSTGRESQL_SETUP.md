# Railway PostgreSQL Database Setup

## ✅ Yes, You Need to Add PostgreSQL Manually

Railway does **NOT** automatically provision a database. You need to add it as a separate service.

## Step-by-Step Setup

### 1. Add PostgreSQL Service

1. **Go to Railway Dashboard**
   - Navigate to your project: `moose-picks-api-production`
   - Click **"+ New"** button (top right)
   - Select **"Database"** → **"Add PostgreSQL"**

2. **PostgreSQL Service Created**
   - Railway will create a new PostgreSQL service
   - It will automatically generate connection credentials
   - The service will appear in your project alongside your API service

### 2. Connect to Your API Service

1. **Select Your API Service**
   - Click on your main API service (the one running FastAPI)

2. **Add Database Variable**
   - Go to **"Variables"** tab
   - Railway should **automatically** add `DATABASE_URL` if services are in the same project
   - If not, click **"+ New Variable"**
   - Name: `DATABASE_URL`
   - Value: Railway will show you the connection string, or you can reference it

3. **Verify Connection String Format**
   - Should look like: `postgresql://postgres:password@hostname:port/railway`
   - Railway handles this automatically when services are linked

### 3. Verify It's Working

1. **Check Environment Variables**
   - In your API service → Variables tab
   - You should see `DATABASE_URL` with a PostgreSQL connection string
   - It should **NOT** be `sqlite:///./moose.db`

2. **Test Connection**
   ```powershell
   Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/health"
   ```
   - Should show `"database": "connected"`
   - Check logs to see if it says `postgresql://` instead of `sqlite://`

3. **Check Logs**
   - Railway Dashboard → Your API Service → Deployments → View Logs
   - Look for: `Database URL: postgresql://...` (not `sqlite://`)

## Current Status Check

Based on your logs, you're currently using **SQLite**:
```
Database URL: sqlite:///./moose.db...
```

After adding PostgreSQL, you should see:
```
Database URL: postgresql://postgres:...
```

## Why PostgreSQL?

- ✅ **Persistent** - Data survives deployments
- ✅ **Scalable** - Handles large datasets (26,000+ games)
- ✅ **Production-ready** - Better for concurrent access
- ✅ **Railway-native** - Managed service with backups

SQLite works locally but:
- ❌ **Ephemeral** - Data lost on Railway deployments
- ❌ **Not scalable** - Single file, limited concurrency
- ❌ **Not production-ready** - File-based, not server-based

## After Adding PostgreSQL

### 1. Redeploy Your API
- Railway will automatically redeploy when `DATABASE_URL` changes
- Or manually trigger: Deployments → Redeploy

### 2. Run Backfill Script
Once PostgreSQL is connected, run the backfill:
```bash
railway run python scripts/backfill_historical_data.py
```

Or via Railway console:
1. Railway Dashboard → Your API Service
2. Use Railway's terminal/console feature
3. Run: `python scripts/backfill_historical_data.py`

### 3. Verify Data is Stored
```powershell
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/health"
```
- `games_in_db` should increase as backfill runs
- Data will persist across deployments

## Troubleshooting

### DATABASE_URL Not Set Automatically

If Railway doesn't auto-add `DATABASE_URL`:

1. **Get Connection String from PostgreSQL Service:**
   - Click on PostgreSQL service
   - Go to "Variables" tab
   - Copy the `DATABASE_URL` value

2. **Add to API Service:**
   - Click on your API service
   - Go to "Variables" tab
   - Click "+ New Variable"
   - Name: `DATABASE_URL`
   - Value: Paste the connection string from step 1

### Still Using SQLite

If logs still show `sqlite://`:

1. **Check Environment Variables:**
   - Make sure `DATABASE_URL` is set in Railway
   - Make sure it's a PostgreSQL URL, not SQLite

2. **Redeploy:**
   - Railway Dashboard → Deployments → Redeploy
   - This ensures new environment variables are loaded

3. **Check Logs:**
   - After redeploy, check startup logs
   - Should show `postgresql://` in database URL

### Connection Errors

If you get database connection errors:

1. **Verify PostgreSQL is Running:**
   - Railway Dashboard → PostgreSQL service
   - Should show "Active" status

2. **Check Connection String:**
   - Make sure `DATABASE_URL` is correct
   - Should start with `postgresql://` or `postgres://`

3. **Check Network:**
   - Services in same project should auto-connect
   - If in different projects, you may need to configure networking

## Quick Checklist

- [ ] Added PostgreSQL service to Railway project
- [ ] `DATABASE_URL` environment variable is set in API service
- [ ] `DATABASE_URL` shows `postgresql://` (not `sqlite://`)
- [ ] API service redeployed after adding database
- [ ] Health check shows `"database": "connected"`
- [ ] Logs show PostgreSQL connection string

## Next Steps

After PostgreSQL is set up:

1. ✅ Run backfill script to populate historical data
2. ✅ Train models (will now have data)
3. ✅ Generate predictions
4. ✅ Set up daily automation

Your data will now persist across deployments! 🎉
