# Trade Republic Knockout Finder

A Selenium-based automation script that logs into Trade Republic's desktop web app and uses binary search through infinite scroll to find knock-out certificates matching a target knockout barrier level.

> **Disclaimer:** Automating Trade Republic's web interface almost certainly violates their Terms of Service. Do not use this code. It is preserved purely as an engineering showcase of creative automation.

> One day, Trade Republic appears to have deliberately disabled infinite scroll on their desktop app, which killed this approach entirely. I was proud to have beaten them for a few weeks, but they won in the end.

---

## The Problem

When the trading system decides to go long on NASDAQ-100 with a knockout barrier at, say, 19,500 points, someone (or something) needs to find the specific knock-out certificate on Trade Republic that has a barrier near that level. Trade Republic's desktop web app lists hundreds of certificates sorted by leverage, loaded progressively through infinite scroll. There is no search-by-knockout-value feature. You scroll, and you scroll, and you scroll.

The certificates are sorted by leverage (descending for longs, ascending for shorts), which means the knockout barrier values are implicitly ordered but not directly searchable. A certificate with higher leverage has a knockout barrier closer to the current price, and vice versa.

---

## The Algorithm

### Step 1: Login

The script uses `undetected_chromedriver` to open Trade Republic's web app, enters the phone number character by character (to avoid bot detection), submits the initial PIN code, then waits for the 2FA code to appear in an S3 file (placed there by a separate process that reads the SMS).

### Step 2: Navigate to Certificates

Navigates to the NASDAQ-100 knockout certificates page, selecting either the long or short tab depending on the ML model's direction signal. The URL encodes the sort order:
- Long: `sort=knockout:desc` (highest knockout first, lowest leverage first)
- Short: `sort=knockout:asc` (lowest knockout first, lowest leverage first)

### Step 3: Binary Search Through Infinite Scroll

This is where it gets interesting. The certificates are loaded lazily as the user scrolls. The script:

1. Scrolls down in 800px increments, waiting 1.5 seconds between scrolls for content to load
2. Parses each visible row to extract leverage, knockout value, and other details
3. Compares each knockout value against the target
4. When the target region is found, clicks the certificate to check its price

The binary search aspect: if a previous attempt found the wrong region, the script remembers the scroll position (`last_scroll_attempts`) and starts the next search from there, effectively narrowing down the scroll range.

### Step 4: Validation and Adjustment

Once a candidate certificate is found:
- If the certificate price is at least 2.00 EUR: success, return the URL for order placement
- If the price is below 2.00 EUR: the knockout is too close to the current price (too much leverage). The script adjusts the target knockout by 10 points further from the current price and retries from the last scroll position
- If leverage exceeds the maximum threshold (default 45x): skip and keep scrolling
- If leverage is anomalously low compared to neighbors (below 70% of the average of adjacent rows): skip, as this likely indicates an untradeable or illiquid certificate

### Step 5: Fallback

If no suitable certificate is found after exhausting the scroll range, the script incrementally relaxes the maximum leverage limit (in steps of 5x, up to 70x) and retries the entire search.

---

## Files

### `trade_traderepublic.py`

The main script containing:
- `login()` -- Full login flow with 2FA via S3
- `find_nearest_knockout()` -- Core scroll-and-search loop
- `find_option()` -- Orchestrator with retry logic and parameter adjustment
- S3 integration for trade signal consumption (`trade.json`) and error logging
- APScheduler-based scheduling for market hours

### `trade_buy.py`

An earlier, simpler version focused only on the buy side of the automation. Preserved for reference.

---

## Technical Details

- Uses `undetected_chromedriver` instead of standard ChromeDriver to avoid Trade Republic's bot detection
- Parses certificate rows from the DOM: `tbody.browseDerivativesLayout__tableBody > tr`
- Handles German number formatting (comma as decimal separator)
- Supports both EUR and point-denominated knockout values
- Filters out futures contracts that appear in the same listing
- Takes screenshots and uploads to S3 for debugging

---

## Why This Exists

Trade Republic does not offer an API for retail users. The only way to programmatically trade knock-out certificates was to automate the web interface. The infinite scroll pattern made this particularly challenging -- unlike a paginated list where you can jump to page N, infinite scroll requires physically scrolling through all preceding content to load the target region.

The binary search optimization reduced average certificate lookup time from several minutes of linear scrolling to under a minute in most cases. It worked for a few glorious weeks until Trade Republic appears to have intentionally removed the infinite scroll from their desktop app, ending the experiment permanently.

**To be absolutely clear:** automating a broker's web interface likely violates their Terms of Service. This code is shared as an engineering curiosity, not as something you should run against your own account.
