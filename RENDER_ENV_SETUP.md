# Render Environment Variables Setup

## Required Environment Variables

Copy these EXACT environment variables into your Render backend service:

### 1. Database Configuration
**Use INTERNAL Database URL (for same-region services):**
```
RIS_POSTGRES_DSN = postgresql://riskxx_sbfh_user:3O0FY2Fso7M6wgF2hKVnSs1NYk3F0hY1@dpg-d4bs1tc9c44c738cmsg0-a/riskxx_sbfh
```

**If internal doesn't work, use EXTERNAL Database URL:**
```
RIS_POSTGRES_DSN = postgresql://riskxx_sbfh_user:3O0FY2Fso7M6wgF2hKVnSs1NYk3F0hY1@dpg-d4bs1tc9c44c738cmsg0-a.oregon-postgres.render.com/riskxx_sbfh
```

### 2. Redis Configuration
```
RIS_REDIS_URL = rediss://default:AYyLAAIncDIxOWY3ZGFkNzYxNzg0MWM0OTQ0NzMwMDUyOTgyZGY1NnAyMzU5Nzk@loyal-arachnid-35979.upstash.io:6379
```

### 3. Environment Settings
```
ENVIRONMENT = production
```

```
RIS_TEST_MODE = false
```

### 4. CORS Configuration
```
RIS_ALLOWED_ORIGINS = https://frontend-1-wvu7.onrender.com,https://app.risksx.io,http://localhost:3000
```

## Step-by-Step Instructions

1. **Go to Render Dashboard**
2. **Find your backend service** (risksx-backend)
3. **Click on "Environment" tab**
4. **Add each environment variable above**:
   - Click "Add Environment Variable"
   - Enter Key (e.g., `RIS_POSTGRES_DSN`)
   - Enter Value (exact string from above)
   - Click "Save Changes"
5. **After adding ALL variables, trigger deployment**:
   - Go to "Settings" tab
   - Click "Manual Deploy"
   - Wait for deployment to complete

## Verification

After deployment, test these URLs:

1. **Environment check**: `https://backend-1-jrik.onrender.com/debug/environment`
   - Should show `postgres_dsn_set: true`, `redis_url_set: true`

2. **Database check**: `https://backend-1-jrik.onrender.com/debug/database`
   - Should show `{"database": "connected", "test_query": 1}`

3. **Router check**: `https://backend-1-jrik.onrender.com/debug/routers`
   - Should show `"loaded_routers": 16` (instead of 11)

4. **GERI endpoint**: `https://backend-1-jrik.onrender.com/api/v1/analytics/geri`
   - Should return GERI data (not "Not Found")

5. **Frontend check**: `https://frontend-1-wvu7.onrender.com`
   - Should stop showing "Loading..." and display actual data

## Troubleshooting

If database still shows "Name or service not known":

1. **Try external hostname** - Check your database service dashboard for external connection string
2. **Check database status** - Ensure database service is "Available" not "Building"
3. **Verify regions** - Database and backend must be in same region
4. **Contact Render support** - Database service might need additional configuration