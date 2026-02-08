
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import json
import traceback
import time as tm
from datetime import datetime
from datetime import timedelta
from pytz import timezone
from apscheduler.schedulers.blocking import BlockingScheduler
import math
import re
import sys
from datetime import datetime, time
# Constants
BUCKET_NAME = 'REDACTED_BUCKET'
TRADE_FILE = 'trade.json'
LOG_FILE = 'error_log.txt'
SECRET_NAME = "BROKER_CREDENTIALS"
REGION_NAME = "eu-central-1"

# Function to retrieve phone and code from AWS Secrets Manager
def get_secret(secret_name, region_name):
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret_dict = json.loads(get_secret_value_response['SecretString'])
        phone = secret_dict.get('phone')
        code = secret_dict.get('code')
        if not phone or not code:
            raise ValueError("Secret must contain 'phone' and 'code' keys")
        print("Successfully retrieved phone and code from Secrets Manager.")
        return phone, code
    except (NoCredentialsError, PartialCredentialsError, ClientError, ValueError) as e:
        print(f"Error retrieving secret: {e}")
        return None, None

# Function to delete a file from S3
def delete_s3_file(bucket_name, file_key):
    s3 = boto3.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=file_key)
        if 'Contents' in response:
            s3.delete_object(Bucket=bucket_name, Key=file_key)
            print(f"File {file_key} deleted from S3 bucket {bucket_name}.")
        else:
            print(f"File {file_key} does not exist in S3 bucket {bucket_name}.")
    except Exception as e:
        print(f"Error deleting {file_key}: {e}")

# Function to read a file from S3 (waits until it exists)
def read_s3_file(bucket_name, file_key):
    s3 = boto3.client('s3')
    while True:
        try:
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=file_key)
            if 'Contents' in response:
                file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
                file_content = file_obj['Body'].read().decode('utf-8')
                print(f"File content: {file_content}")
                return file_content
            print(f"File {file_key} does not exist. Waiting 10 seconds...")
            tm.sleep(10)
        except Exception as e:
            print(f"Error reading {file_key}: {e}")
            break

# Function to check and delete trade.json
def check_and_delete_trade_json(bucket_name, file_name='trade.json'):
    s3 = boto3.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=file_name)
        if 'Contents' in response:
            print(f"{file_name} exists in the bucket {bucket_name}.")
            obj = s3.get_object(Bucket=bucket_name, Key=file_name)
            data = obj['Body'].read().decode('utf-8')
            trade_data = json.loads(data)
            print(f"Contents of {file_name}: {trade_data}")
            s3.delete_object(Bucket=bucket_name, Key=file_name)  # Uncommented as requested
            print(f"{file_name} has been deleted from the bucket {bucket_name}.")
            return trade_data
        print(f"{file_name} does not exist in the bucket {bucket_name}.")
        return None
    except Exception as e:
        print(f"Error checking/deleting {file_name}: {e}")
        return None

# Login function
import time as tm
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
import re

# Existing imports and constants (assumed)
# BUCKET_NAME, LOGIN_FILE, SECRET_NAME, REGION_NAME, get_secret, delete_s3_file, read_s3_file

def login():
    phone, code = get_secret(SECRET_NAME, REGION_NAME)
    if not phone or not code:
        return None

    options = Options()
    options.add_argument("start-maximized")
    options.add_argument("--window-size=1920,1080")

    driver = uc.Chrome(options=options, use_subprocess=False)
    driver.get("https://google.com")
    tm.sleep(5)

    driver.get("https://app.traderepublic.com/login")
    driver.implicitly_wait(5)
    tm.sleep(5)

    # Accept cookies
    try:
        button = driver.find_element(By.XPATH, "//button[@class='buttonBase consentCard__action buttonPrimary consentCard__action']//span[text()='Accept All']")
        button.click()
    except Exception as e:
        print(f"Warning: Could not accept cookies: {e}")
    tm.sleep(0.5)
    # Enter phone number
    input_element = driver.find_element(By.ID, 'loginPhoneNumber__input')
    for char in phone:
        input_element.send_keys(char)
        tm.sleep(0.5)
    input_element.send_keys(Keys.RETURN)
    tm.sleep(1)
    # Delete login.txt from S3 before requesting 2FA code
    delete_s3_file(BUCKET_NAME, 'login.txt')
    # Enter initial code
    actions = ActionChains(driver)
    for char in code:
        actions.send_keys(char).perform()
        tm.sleep(0.5)
    actions.send_keys(Keys.RETURN).perform()
    tm.sleep(1)

    

    # Wait for and read the new 2FA code from S3
    two_factor_code = read_s3_file(BUCKET_NAME, 'login.txt')
    if not two_factor_code:
        print("Failed to retrieve 2FA code. Exiting login.")
        driver.quit()
        return None

    # Enter the 2FA code
    actions.send_keys(two_factor_code + Keys.RETURN).perform()
    tm.sleep(5)

    # Handle "Got it" button if it appears
    try:
        ok_button = driver.find_element(By.XPATH, "//span[@class='buttonBase__title' and text()='Got it']")
        ok_button.click()
    except Exception:
        pass

    return driver

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def find_nearest_knockout(searched_knockout, direction, driver, max_leverage=45, max_scroll_attempts=100, leverage_anomaly_threshold=0.7, initial_scroll_attempts=0):
    """
    Find a knockout option that meets the criteria, scrolling to load all options if needed.
    Skips options with abnormally low leverage (potential untradeable options).

    Args:
        searched_knockout (float): The knockout value to search for.
        direction (int): 1 for Long, -1 for Short.
        driver: WebDriver instance.
        max_leverage (float): Maximum allowed leverage (default: 45).
        max_scroll_attempts (int): Maximum number of scroll attempts (default: 100).
        leverage_anomaly_threshold (float): Threshold for identifying untradeable options (default: 0.7).
        initial_scroll_attempts (int): Number of scroll attempts to jump to initially (default: 0).

    Returns:
        tuple: (result, scroll_attempts)
            - result: True if price >= 2.0, None if price < 2.0, False if no option found.
            - scroll_attempts: Number of scroll attempts at the point of result.
    """
    try:
        scroll_attempts = 0
        processed_rows = set()

        # Jump to the initial scroll position if provided
        if initial_scroll_attempts > 0:
            driver.execute_script(f"window.scrollBy(0, {800 * initial_scroll_attempts});")
            scroll_attempts = initial_scroll_attempts
            print(f"Jumped to scroll position {800 * initial_scroll_attempts}px (attempt {scroll_attempts})")
            tm.sleep(1.5)  # Wait for content to load

        while scroll_attempts < max_scroll_attempts:
            rows = driver.find_elements(By.XPATH, "//tbody[@class='browseDerivativesLayout__tableBody']/tr[not(@aria-hidden='true')]")
            for i, row in enumerate(rows):
                row_text = row.text
                if row_text in processed_rows:
                    continue
                processed_rows.add(row_text)
                if "future" in row_text.lower():
                    continue
                parts = row_text.split('\n')
                try:
                    leverage = float(parts[0].replace('x', '').strip())
                    # Check for leverage anomalies
                    neighbor_avg = 0
                    if i > 0:
                        prev_leverage = float(rows[i-1].text.split('\n')[0].replace('x', '').strip())
                        neighbor_avg += prev_leverage
                    if i < len(rows) - 1:
                        next_leverage = float(rows[i+1].text.split('\n')[0].replace('x', '').strip())
                        neighbor_avg += next_leverage
                    neighbor_avg = neighbor_avg / (1 if i == 0 or i == len(rows) - 1 else 2)

                    if neighbor_avg > 0 and leverage < neighbor_avg * leverage_anomaly_threshold:
                        print(f"Skipping row {i}: Leverage {leverage}x is an anomaly (< {leverage_anomaly_threshold*100}% of neighbor average {neighbor_avg}x)")
                        continue
                    if leverage > max_leverage:
                        print(f"Skipping row {i}: Leverage {leverage}x exceeds max_leverage {max_leverage}")
                        continue

                    value = float(parts[2].replace(' Pkt.', '').replace(' $', '').replace(' €', '').replace(' Pts.', '').replace(',', ''))
                    if (value - searched_knockout) * direction < -10:
                        row.click()
                        tm.sleep(3)
                        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//span[@role="status"]')))
                        price_element = driver.find_element(By.XPATH, '//span[@role="status"]')
                        price_value = float(price_element.text.replace('€', '').strip())
                        if price_value >= 2.0:
                            print(f"Found option: price {price_value}€, leverage {leverage}x, knockout {value}")
                            return (True, scroll_attempts)
                        else:
                            print(f"Price {price_value}€ < 2.0 for knockout {value}. Need adjustment.")
                            driver.back()
                            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, "//tbody[@class='browseDerivativesLayout__tableBody']/tr[not(@aria-hidden='true')]")))
                            return (None, scroll_attempts)
                except Exception as e:
                    print(f"Error processing row {i}: {e}")
                    continue

            driver.execute_script("window.scrollBy(0, 800);")
            print(f"Scrolled down 800px (attempt {scroll_attempts + 1}/{max_scroll_attempts})")
            tm.sleep(1.5)
            scroll_attempts += 1

        print(f"No suitable option found for knockout {searched_knockout} after {max_scroll_attempts} attempts")
        return (False, scroll_attempts)

    except Exception as e:
        print(f"Error in find_nearest_knockout: {e}")
        return (False, scroll_attempts)
    
def find_option(knockout_input, direction, driver, max_attempts=10, max_leverage=45, leverage_increment=5, max_leverage_limit=70, leverage_anomaly_threshold=0.7):
    """
    Find a suitable knockout option, adjusting the knockout value and leverage if necessary.

    Args:
        knockout_input (float): Initial knockout value.
        direction (int): 1 for Long, -1 for Short.
        driver: WebDriver instance.
        max_attempts (int): Max number of knockout adjustments (default: 10).
        max_leverage (float): Initial max leverage (default: 45).
        leverage_increment (float): Amount to increase max_leverage if needed (default: 5).
        max_leverage_limit (float): Maximum allowable leverage (default: 70).
        leverage_anomaly_threshold (float): Threshold for identifying untradeable options (default: 0.7).

    Returns:
        str: Current URL if successful, None otherwise.
    """
    knockout = knockout_input
    current_max_leverage = max_leverage
    url = (
        "https://app.traderepublic.com/derivatives/US6311011026/knockouts?tab=long&sort=knockout:desc"
        if direction == 1
        else "https://app.traderepublic.com/derivatives/US6311011026/knockouts?tab=short&sort=knockout:asc"
    )
    driver.get(url)
    print(f"Navigated to {url}")

    attempt = 0
    last_scroll_attempts = 0  # Store the last scroll position

    while attempt < max_attempts:
        print(f"Attempt {attempt + 1}/{max_attempts} with knockout {knockout}, max_leverage {current_max_leverage}")
        result, scroll_attempts = find_nearest_knockout(
            knockout, direction, driver, max_leverage=current_max_leverage,
            leverage_anomaly_threshold=leverage_anomaly_threshold, initial_scroll_attempts=last_scroll_attempts
        )
        last_scroll_attempts = scroll_attempts  # Update the last scroll position

        if result is True:
            print("Success! Returning current URL.")
            return driver.current_url
        elif result is None:
            attempt += 1
            knockout = knockout - 10 if direction == 1 else knockout + 10
            print(f"Price too low. Adjusted knockout to {knockout}")
            tm.sleep(1)
            # Next iteration will start from last_scroll_attempts
        else:  # result is False
            if current_max_leverage >= max_leverage_limit:
                print(f"No suitable options found after reaching max leverage limit {max_leverage_limit}.")
                return None
            current_max_leverage += leverage_increment
            print(f"No options found within leverage {current_max_leverage - leverage_increment}. Increased max_leverage to {current_max_leverage}")
            # Reset scroll position since leverage change may require re-evaluating earlier options
            last_scroll_attempts = 0
            driver.get(url)  # Reload the page to start from the top

    print(f"Failed to find a suitable option after {max_attempts} attempts for knockout range starting at {knockout_input}.")
    return None

def get_visible_knockouts(driver):
    """Extract knockout values from currently visible rows."""
    try:
        rows = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located(
                (By.XPATH, "//tbody[@class='browseDerivativesLayout__tableBody']/tr[not(@aria-hidden='true')]")
            )
        )
        knockouts = []
        for row in rows:
            try:
                parts = row.text.split('\n')
                value = float(parts[2].replace(' Pkt.', '').replace(' $', '').replace(' €', '').replace(' Pts.', '').replace(',', ''))
                knockouts.append(value)
            except Exception:
                continue
        return knockouts
    except Exception as e:
        print(f"Error in get_visible_knockouts: {e}")
        return []

def find_nearest_knockout_optimized(searched_knockout, direction, driver, max_leverage=45, max_scroll_attempts=100, leverage_anomaly_threshold=0.7, initial_scroll_attempts=0):
    """
    Find a knockout option using an optimized scrolling strategy: coarse initial scrolls with binary adjustment, then fine-tuning.
    
    Args:
        searched_knockout (float): The knockout value to search for.
        direction (int): 1 for Long, -1 for Short.
        driver: WebDriver instance.
        max_leverage (float): Maximum allowed leverage (default: 45).
        max_scroll_attempts (int): Maximum number of scroll attempts (default: 100).
        leverage_anomaly_threshold (float): Threshold for identifying untradeable options (default: 0.7).
        initial_scroll_attempts (int): Number of scroll attempts to jump to initially (default: 0).

    Returns:
        tuple: (result, scroll_attempts)
            - result: True if price >= 2.0, None if price < 2.0, False if no option found.
            - scroll_attempts: Number of scroll attempts at the point of result.
    """
    try:
        scroll_attempts = 0
        processed_rows = set()
        current_scroll = 0
        max_stagnant_attempts = 5  # Stop after 5 scrolls with no new rows
        stagnant_count = 0
        previous_row_count = 0

        if initial_scroll_attempts == 0:
            # Coarse scrolling phase with binary-like adjustment
            low = 0
            high = 100000  # Arbitrary max scroll height
            scroll_step = 10000  # Large initial step

            while low <= high:
                mid = (low + high) // 2
                driver.execute_script(f"window.scrollTo(0, {mid});")
                tm.sleep(2)  # Increased wait for content to load
                knockouts = get_visible_knockouts(driver)

                if not knockouts:
                    high = mid - 1
                    continue

                if direction == 1:  # Long: target is where knockout < searched_knockout - 10
                    if min(knockouts) >= searched_knockout - 10:
                        low = mid + 1  # Target is further down
                    else:
                        high = mid - 1  # Overshot, target is above
                else:  # Short: target is where knockout > searched_knockout + 10
                    if max(knockouts) <= searched_knockout + 10:
                        low = mid + 1  # Target is further down
                    else:
                        high = mid - 1  # Overshot, target is above

                if high - low < scroll_step:  # Close enough to fine-tune
                    break

            current_scroll = (low + high) // 2
            driver.execute_script(f"window.scrollTo(0, {current_scroll});")
            tm.sleep(2)
            print(f"Coarse scrolling positioned at {current_scroll}px")
            scroll_attempts = current_scroll // 800  # Approximate attempts
        else:
            # Jump to last known position
            current_scroll = 800 * initial_scroll_attempts
            driver.execute_script(f"window.scrollTo(0, {current_scroll});")
            scroll_attempts = initial_scroll_attempts
            tm.sleep(2)
            print(f"Jumped to scroll position {current_scroll}px (attempt {scroll_attempts})")

        # Fine-tuning with incremental scrolling
        while scroll_attempts < max_scroll_attempts:
            rows = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located(
                    (By.XPATH, "//tbody[@class='browseDerivativesLayout__tableBody']/tr[not(@aria-hidden='true')]")
                )
            )
            current_row_count = len(rows)

            # Check for end of list
            if current_row_count == previous_row_count:
                stagnant_count += 1
                if stagnant_count >= max_stagnant_attempts:
                    print(f"No new rows loaded after {max_stagnant_attempts} attempts. Assuming end of list.")
                    return (False, scroll_attempts)
            else:
                stagnant_count = 0
            previous_row_count = current_row_count

            for i, row in enumerate(rows):
                try:
                    row_text = row.text
                    if row_text in processed_rows:
                        continue
                    processed_rows.add(row_text)
                    if "future" in row_text.lower():
                        continue
                    parts = row_text.split('\n')
                    leverage = float(parts[0].replace('x', '').strip())

                    # Check leverage anomaly
                    neighbor_avg = 0
                    if i > 0:
                        prev_leverage = float(rows[i-1].text.split('\n')[0].replace('x', '').strip())
                        neighbor_avg += prev_leverage
                    if i < len(rows) - 1:
                        next_leverage = float(rows[i+1].text.split('\n')[0].replace('x', '').strip())
                        neighbor_avg += next_leverage
                    neighbor_avg = neighbor_avg / (1 if i == 0 or i == len(rows) - 1 else 2)

                    if neighbor_avg > 0 and leverage < neighbor_avg * leverage_anomaly_threshold:
                        print(f"Skipping row {i}: Leverage {leverage}x is an anomaly (< {leverage_anomaly_threshold*100}% of neighbor average {neighbor_avg}x)")
                        continue
                    if leverage > max_leverage:
                        print(f"Skipping row {i}: Leverage {leverage}x exceeds max_leverage {max_leverage}")
                        continue

                    value = float(parts[2].replace(' Pkt.', '').replace(' $', '').replace(' €', '').replace(' Pts.', '').replace(',', ''))
                    if (value - searched_knockout) * direction < -10:
                        # Ensure element is clickable
                        row = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.XPATH, f"//tbody[@class='browseDerivativesLayout__tableBody']/tr[not(@aria-hidden='true')][{i+1}]"))
                        )
                        row.click()
                        tm.sleep(1)  # Short wait before checking price
                        price_element = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, '//span[@role="status"]'))
                        )
                        price_value = float(price_element.text.replace('€', '').strip())
                        if price_value >= 2.0:
                            print(f"Found option: price {price_value}€, leverage {leverage}x, knockout {value}")
                            return (True, scroll_attempts)
                        else:
                            print(f"Price {price_value}€ < 2.0 for knockout {value}. Need adjustment.")
                            driver.back()
                            WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located(
                                    (By.XPATH, "//tbody[@class='browseDerivativesLayout__tableBody']/tr[not(@aria-hidden='true')]")
                                )
                            )
                            return (None, scroll_attempts)
                except StaleElementReferenceException:
                    print(f"Stale element detected at row {i}. Re-fetching rows.")
                    break  # Break inner loop to re-fetch rows
                except Exception as e:
                    print(f"Error processing row {i}: {e}")
                    continue

            driver.execute_script("window.scrollBy(0, 800);")
            current_scroll += 800
            tm.sleep(2)  # Increased wait for content to load
            scroll_attempts += 1
            print(f"Scrolled down 800px to {current_scroll}px (attempt {scroll_attempts}/{max_scroll_attempts})")

        print(f"No suitable option found for knockout {searched_knockout} after {max_scroll_attempts} attempts")
        return (False, scroll_attempts)

    except Exception as e:
        print(f"Error in find_nearest_knockout_optimized: {e}")
        return (False, scroll_attempts)

def find_option_optimized(knockout_input, direction, driver, max_attempts=10, max_leverage=45, leverage_increment=5, max_leverage_limit=70, leverage_anomaly_threshold=0.7):
    """
    Find a suitable knockout option with optimized scrolling, adjusting knockout and leverage as needed.

    Args:
        knockout_input (float): Initial knockout value.
        direction (int): 1 for Long, -1 for Short.
        driver: WebDriver instance.
        max_attempts (int): Max number of knockout adjustments (default: 10).
        max_leverage (float): Initial max leverage (default: 45).
        leverage_increment (float): Amount to increase max_leverage if needed (default: 5).
        max_leverage_limit (float): Maximum allowable leverage (default: 70).
        leverage_anomaly_threshold (float): Threshold for identifying untradeable options (default: 0.7).

    Returns:
        str: Current URL if successful, None otherwise.
    """
    knockout = knockout_input
    current_max_leverage = max_leverage
    url = (
        "https://app.traderepublic.com/derivatives/US6311011026/knockouts?tab=long&sort=knockout:desc"
        if direction == 1
        else "https://app.traderepublic.com/derivatives/US6311011026/knockouts?tab=short&sort=knockout:asc"
    )
    driver.get(url)
    print(f"Navigated to {url}")

    attempt = 0
    last_scroll_attempts = 0

    while attempt < max_attempts:
        print(f"Attempt {attempt + 1}/{max_attempts} with knockout {knockout}, max_leverage {current_max_leverage}")
        result, scroll_attempts = find_nearest_knockout_optimized(
            knockout, direction, driver, max_leverage=current_max_leverage,
            leverage_anomaly_threshold=leverage_anomaly_threshold, initial_scroll_attempts=last_scroll_attempts
        )
        last_scroll_attempts = scroll_attempts

        if result is True:
            print("Success! Returning current URL.")
            return driver.current_url
        elif result is None:
            attempt += 1
            knockout = knockout - 10 if direction == 1 else knockout + 10
            print(f"Price too low. Adjusted knockout to {knockout}")
            tm.sleep(1)
        else:
            if current_max_leverage >= max_leverage_limit:
                print(f"No suitable options found after reaching max leverage limit {max_leverage_limit}.")
                return None
            current_max_leverage += leverage_increment
            print(f"No options found within leverage {current_max_leverage - leverage_increment}. Increased max_leverage to {current_max_leverage}")
            last_scroll_attempts = 0
            driver.get(url)

    print(f"Failed to find a suitable option after {max_attempts} attempts for knockout range starting at {knockout_input}.")
    return None

def switch_to_limit_order(driver):
    try:
        menu_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, '//button[contains(@class, "dropdownList__openButton")]')))
        menu_button.click()
        limit_option = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, '//li[@id="orderBegin__modeSelection-limit"]')))
        limit_option.click()
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//li[@id="orderBegin__modeSelection-limit" and contains(@class, "-selected")]')))
        
    except Exception as e:
        print(f"Error switching to limit order: {e}")
        

def set_order_details(driver, trade_amount=None):
    try:
        WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CLASS_NAME, 'orderFlowCard')))
        price_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//span[@role="status"]')))
        current_price = float(price_element.text.replace('€', '').strip())
        limit_price = current_price + 0.5
        funds_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'orderFormAvailableFunds')))
        available_funds = float(re.sub(r'[^\d.]', '', funds_element.find_element(By.TAG_NAME, 'p').text.strip()))
        print(f"Available funds: {available_funds}€")
        available_funds_with_commission = available_funds - 1.0
        if available_funds_with_commission <= 0:
            print(f"Error: Insufficient funds after commission: {available_funds}€")
            
        trade_amount = available_funds_with_commission if trade_amount is None else trade_amount
        print(f"Using trade amount: {trade_amount}€")
        max_shares = math.floor(available_funds_with_commission / limit_price)
        num_shares = min(max_shares, math.floor(trade_amount / current_price))
        if num_shares < 1:
            print(f"Error: Cannot afford any shares with {trade_amount}€ and {available_funds}€ available")
            
        shares_input = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'orderFlow__limitbuyShares')))
        price_input = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'orderFlow__limitbuyPrice')))
        driver.execute_script("arguments[0].click();", shares_input)
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, 'orderFlow__limitbuyShares')))
        actions = ActionChains(driver)
        actions.move_to_element(shares_input).click().send_keys(str(num_shares)).perform()
        driver.execute_script("arguments[0].click();", price_input)
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, 'orderFlow__limitbuyPrice')))
        actions.move_to_element(price_input).click().send_keys(f"{limit_price:.2f}").perform()
        tm.sleep(0.5)
        shares_value = driver.execute_script("return arguments[0].value;", shares_input)
        price_value = driver.execute_script("return arguments[0].value;", price_input)
            
        print(f"Set {num_shares} shares at {limit_price:.2f}€ for {trade_amount}€ (within {available_funds}€, after 1€ commission)")
        
    except Exception as e:
        print(f"Error setting order details: {e}")
        

def review_order(driver):
    try:
        WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CLASS_NAME, 'orderFlowCard')))
        review_span = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//span[@class="buttonBase__title" and text()="Review Order"]')))
        review_button = review_span.find_element(By.XPATH, './ancestor::button')
        if review_button.get_attribute('disabled') or 'disabled' in review_button.get_attribute('class'):
            print("Error: Review Order button is disabled")
            
        driver.execute_script("arguments[0].click();", review_button)
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'orderFlowCard')))
        print("Successfully clicked Review Order")
        
    except Exception as e:
        print(f"Error clicking Review Order: {e}")
        

def buy_now(driver):
    try:
        buy_span = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//span[@class="buttonBase__title" and text()="Buy now "]')))
        buy_button = buy_span.find_element(By.XPATH, './ancestor::button')
        if buy_button.get_attribute('disabled') or 'disabled' in buy_button.get_attribute('class'):
            print("Error: Buy now button is disabled")
            
        if not buy_button.is_displayed():
            print("Error: Buy now button is not visible")
            
        driver.execute_script("arguments[0].click();", buy_button)
        try:
            WebDriverWait(driver, 5).until(EC.invisibility_of_element_located((By.CLASS_NAME, 'overlay__content')))
            print("Order dialog closed, buy action likely completed")
        except:
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//p[contains(text(), "Order placed successfully")]')))
            print("Confirmation message found")
        print("Successfully clicked Buy now")
        
    except Exception as e:
        print(f"Error clicking Buy now: {e}")
        

def reload_page(driver):
    try:
        current_url = driver.current_url
        print(f"Reloading page from URL: {current_url}")
        driver.refresh()
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'layout__secondaryContent')))
        print("Page reloaded successfully")
        
    except Exception as e:
        print(f"Error reloading page: {e}")
        

def click_sell(driver):
    try:
        WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CLASS_NAME, 'orderFlowCard')))
        sell_tab = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'orderFlowTabs__sellTab')))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", sell_tab)
        WebDriverWait(driver, 5).until(EC.visibility_of(sell_tab))
        if sell_tab.get_attribute('disabled') or 'disabled' in sell_tab.get_attribute('class'):
            print("Error: Sell tab button is disabled")
            
        if not sell_tab.is_displayed():
            print("Error: Sell tab button is not visible after scrolling")
            
        driver.execute_script("arguments[0].click();", sell_tab)
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//button[@id="orderFlowTabs__sellTab" and @aria-selected="true"]')))
        print("Successfully clicked Sell tab button")
        
    except Exception as e:
        print(f"Error clicking Sell tab button: {e}")
        

def switch_to_stop_order(driver):
    try:
        WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CLASS_NAME, 'orderFlowCard')))
        dropdown_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, '//button[@class="dropdownList__openButton" and @aria-label="Select Order Type"]')))
        driver.execute_script("arguments[0].click();", dropdown_button)
        tm.sleep(1)
        WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CLASS_NAME, 'dropdownList__listbox')))
        actions = ActionChains(driver)
        actions.send_keys(Keys.DOWN).send_keys(Keys.DOWN).send_keys(Keys.ENTER).perform()
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//li[@id="orderBegin__modeSelection-stopMarket" and contains(@class, "-selected")]')))
        print("Successfully switched to Stop Order")
        return True
    except Exception as e:
        print(f"Error switching to Stop Order: {e}")
        return False
        

def set_stop_order_details(driver):
    try:
        WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CLASS_NAME, 'orderFlowCard')))
        shares_available_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'orderFormAvailableFunds')))
        num_shares = int(re.search(r'\d+', shares_available_element.find_element(By.TAG_NAME, 'p').text.strip()).group())
        buy_in_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'instrumentPosition__buyIn')))
        buy_in_price = float(buy_in_element.text.strip().replace('€', '').replace(',', '.'))
        stop_price = round(buy_in_price * 0.78, 2)
        shares_input = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'orderFlow__stopMarketsellShares')))
        stop_price_input = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'orderFlow__stopMarketsellPrice')))
        driver.execute_script("arguments[0].click();", shares_input)
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, 'orderFlow__stopMarketsellShares')))
        actions = ActionChains(driver)
        actions.move_to_element(shares_input).click().send_keys(str(num_shares)).perform()
        driver.execute_script("arguments[0].click();", stop_price_input)
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, 'orderFlow__stopMarketsellPrice')))
        actions.move_to_element(stop_price_input).click().send_keys(str(stop_price)).perform()
        tm.sleep(0.5)
        shares_value = driver.execute_script("return arguments[0].value;", shares_input)
        stop_price_value = driver.execute_script("return arguments[0].value;", stop_price_input)
        print(f"Set {num_shares} shares with stop price {stop_price}€ (based on buy-in price {buy_in_price}€)")
        
    except Exception as e:
        print(f"Error setting stop order details: {e}")
        

def sell_now(driver):
    try:
        sell_span = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//span[@class="buttonBase__title" and text()="Sell now "]')))
        sell_button = sell_span.find_element(By.XPATH, './ancestor::button')
        if sell_button.get_attribute('disabled') or 'disabled' in sell_button.get_attribute('class'):
            print("Error: Sell now button is disabled")
            
        if not sell_button.is_displayed():
            print("Error: Sell now button is not visible")
            
        driver.execute_script("arguments[0].click();", sell_button)
        try:
            WebDriverWait(driver, 5).until(EC.invisibility_of_element_located((By.CLASS_NAME, 'overlay__content')))
            print("Order dialog closed, sell action likely completed")
        except:
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//p[contains(text(), "Order placed successfully")]')))
            print("Confirmation message found")
        print("Successfully clicked Sell now")
        
    except Exception as e:
        print(f"Error clicking Sell now: {e}")
        

def click_and_press_down(driver, element_class="instrumentWarningMessage", num_presses=7):
    try:
        element = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.CLASS_NAME, element_class)))
        if not element.is_displayed():
            print(f"Error: Element with class '{element_class}' is not visible")
            
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
        element.click()
        actions = ActionChains(driver)
        actions.move_to_element(element).click()
        for _ in range(num_presses):
            actions.send_keys(Keys.DOWN).perform()
        WebDriverWait(driver, 5).until(lambda driver: driver.execute_script("return document.readyState;") == "complete")
        print(f"Successfully clicked element '{element_class}' and pressed Down key {num_presses} times")
        
    except Exception as e:
        print(f"Error clicking and pressing Down on element '{element_class}': {e}")
        

def navigate_to_stop_market_orders(driver):
    try:
        # Navigate to the specified URL
        target_url = "https://app.traderepublic.com/orders/stopMarket"
        driver.get(target_url)
        print(f"Navigating to: {target_url}")

        # Wait for the page to load (adjust the locator based on the target page's structure)
        # Using a generic class or ID that might exist on the orders/stopMarket page
        # Replace 'ordersPage' with a specific class or ID from the target page if known
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'ordersPage'))  # Placeholder; adjust as needed
        )

        print("Successfully navigated to orders/stopMarket page")
        return True

    except Exception as e:
        print(f"Error navigating to orders/stopMarket: {str(e)}")
        return False
    


def click_first_order(driver):
    try:
        # Ensure the stop market orders tab panel is visible
        WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.ID, 'ordersTabPanels__stopMarketTabPanel'))
        )

        # Find the first order in the order list
        first_order = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((
                By.XPATH, 
                '//ol[@class="orderList instrumentOrders__list"]/li[1]'
            ))
        )

        # Scroll to the order to ensure it's in the viewport
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", first_order)

        # Wait for the order to be visible
        WebDriverWait(driver, 5).until(
            EC.visibility_of(first_order)
        )

        # Ensure the order is not disabled (though the HTML doesn't indicate a disabled state)
        if first_order.get_attribute('disabled'):
            print("Error: First order is disabled")
            return False

        # Double-check visibility
        if not first_order.is_displayed():
            print("Error: First order is not visible after scrolling")
            return False

        # Use JavaScript to click the order
        driver.execute_script("arguments[0].click();", first_order)

        # Wait for a potential page transition or detail view (adjust based on what happens after click)
        # For now, wait for the order panel to remain or a detail element to appear
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.ID, 'ordersTabPanels__stopMarketTabPanel'))  # Check if still on the same panel
        )

        print("Successfully clicked the first order")
        return True

    except Exception as e:
        print(f"Error clicking the first order: {str(e)}")
        return False

def click_delete_button(driver):
    try:
        # Ensure the side modal is visible
        WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.CLASS_NAME, 'sideModalLayout__content'))
        )

        # Find the "Delete" button by locating the span and getting its parent button
        delete_span = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, '//span[@class="buttonBase__title" and text()="Delete"]'))
        )
        delete_button = delete_span.find_element(By.XPATH, './ancestor::button')

        # Scroll to the button to ensure it's in the viewport
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", delete_button)

        # Wait for the button to be visible
        WebDriverWait(driver, 5).until(
            EC.visibility_of(delete_button)
        )

        # Ensure the button is not disabled
        if delete_button.get_attribute('disabled') or 'disabled' in delete_button.get_attribute('class'):
            print("Error: Delete button is disabled")
            return False

        # Double-check visibility
        if not delete_button.is_displayed():
            print("Error: Delete button is not visible after scrolling")
            return False

        # Use JavaScript to click the button
        driver.execute_script("arguments[0].click();", delete_button)

        # Wait for a potential confirmation or the modal to close (adjust based on behavior)
        # For now, wait for the side modal to disappear or a confirmation message
        try:
            WebDriverWait(driver, 5).until(
                EC.invisibility_of_element_located((By.CLASS_NAME, 'sideModalLayout__content'))
            )
            print("Side modal closed, delete action likely completed")
        except:
            # Alternatively, wait for a confirmation message (adjust XPath if needed)
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//p[contains(text(), "Order deleted successfully")]'))
            )
            print("Confirmation message found")

        print("Successfully clicked Delete button")
        return True

    except Exception as e:
        print(f"Error clicking Delete button: {str(e)}")
        return False
    

def confirm_delete_order(driver):
    try:
        # Ensure the confirmation dialog is visible
        WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.CLASS_NAME, 'orderDelete'))
        )

        # Find the "Delete Order" button by locating the span and getting its parent button
        delete_order_span = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, '//span[@class="buttonBase__title" and text()="Delete Order"]'))
        )
        delete_order_button = delete_order_span.find_element(By.XPATH, './ancestor::button')

        # Scroll to the button to ensure it's in the viewport
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", delete_order_button)

        # Wait for the button to be visible
        WebDriverWait(driver, 5).until(
            EC.visibility_of(delete_order_button)
        )

        # Ensure the button is not disabled
        if delete_order_button.get_attribute('disabled') or 'disabled' in delete_order_button.get_attribute('class'):
            print("Error: Delete Order button is disabled")
            return False

        # Double-check visibility
        if not delete_order_button.is_displayed():
            print("Error: Delete Order button is not visible after scrolling")
            return False

        # Use JavaScript to click the button
        driver.execute_script("arguments[0].click();", delete_order_button)

        # Wait for the confirmation dialog to close or a success message
        try:
            WebDriverWait(driver, 5).until(
                EC.invisibility_of_element_located((By.CLASS_NAME, 'orderDelete'))
            )
            print("Confirmation dialog closed, order deleted successfully")
        except:
            # Alternatively, wait for a success message (adjust XPath if needed)
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//p[contains(text(), "Order deleted successfully")]'))
            )
            print("Order deletion confirmation message found")

        print("Successfully confirmed deletion of order")
        return True

    except Exception as e:
        print(f"Error confirming deletion of order: {str(e)}")
        return False
    
def set_market_order_details(driver):
    try:
        # Ensure the order form dialog is visible
        WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.CLASS_NAME, 'orderFlowCard'))
        )

        # Verify that Market order is selected in the dropdown
        market_order_selected = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((
                By.XPATH, 
                '//li[@id="orderBegin__modeSelection-market" and contains(@class, "-selected")]'
            ))
        )
        if not market_order_selected:
            print("Error: Market order is not selected in the dropdown")
            return False

        # Get the number of shares available
        shares_available_element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'orderFormAvailableFunds'))
        )
        shares_text = shares_available_element.find_element(By.TAG_NAME, 'p').text.strip()
        num_shares = int(re.search(r'\d+', shares_text).group())  # Extract the number (e.g., 2)

        # Find the shares input field
        shares_input = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.ID, 'orderFlow__marketsellShares'))
        )

        # Use JavaScript to click the shares input
        driver.execute_script("arguments[0].click();", shares_input)

        # Wait for the input to be clickable
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, 'orderFlow__marketsellShares'))
        )

        # Use ActionChains to type the number of shares
        actions = ActionChains(driver)
        actions.move_to_element(shares_input).click().send_keys(str(num_shares)).perform()

        # Small delay to allow form processing
        tm.sleep(0.5)

        print(f"Set market order to sell {num_shares} shares")
        return True

    except Exception as e:
        print(f"Error setting market order details: {str(e)}")
        return False





def get_current_url(driver):
    try:
        current_url = driver.current_url
        print(f"Current URL: {current_url}")
        return current_url
    except Exception as e:
        print(f"Error getting current URL: {e}")
        return None

url = None

def sell_option(option_url):
    driver = login()
    tm.sleep(5)
    navigate_to_stop_market_orders(driver)
    tm.sleep(5)
    click_first_order(driver)
    tm.sleep(5)
    click_delete_button(driver)
    tm.sleep(5)
    confirm_delete_order(driver)
    driver.get(option_url)
    tm.sleep(5)
    click_sell(driver)
    tm.sleep(5)
    try:
        click_and_press_down(driver)
    except Exception as e:
        print(f"Warning: Failed to press Down keys: {e}")
    set_market_order_details(driver)
    tm.sleep(5)
    review_order(driver)
    tm.sleep(2)
    
    sell_now(driver)

    driver.quit()


def start_buy(knockout, direction):
    global url
    driver = login()
    tm.sleep(5)
    url = find_option(knockout, direction, driver)
    
    tm.sleep(5)
    switch_to_limit_order(driver)
      
    tm.sleep(2)
    set_order_details(driver)
       
    tm.sleep(2)
    try:
        click_and_press_down(driver)
    except Exception as e:
        print(f"Warning: Failed to press Down keys: {e}")
    tm.sleep(2)
    review_order(driver)
    
    tm.sleep(2)
   
    buy_now(driver)
        
    tm.sleep(2)
    
    succes = False
    while not succes:
        reload_page(driver)
        tm.sleep(2)
        click_sell(driver)
        tm.sleep(4)
        tm.sleep(2)
        succes = switch_to_stop_order(driver)
        tm.sleep(2)
         

        
    tm.sleep(2)
    set_stop_order_details(driver)
        
    tm.sleep(2)
    try:
        click_and_press_down(driver)
    except Exception as e:
        print(f"Warning: Failed to press Down keys: {e}")
    tm.sleep(2)
    review_order(driver)
        
    tm.sleep(2)
    
    sell_now(driver)

    driver.quit()
     

def log_error_to_s3(error_traceback, bucket_name, file_name='error_log.txt'):
    s3 = boto3.client('s3')
    try:
        timestamp = datetime.now(timezone('UTC')).strftime('%Y-%m-%d_%H-%M-%S')
        log_entry = f"Error at {timestamp}:\n{error_traceback}\n"
        s3.put_object(Bucket=bucket_name, Key=file_name, Body=log_entry.encode('utf-8'))
        print(f"Error logged to S3 bucket {bucket_name} as {file_name}")
    except Exception as e:
        print(f"Failed to log error to S3: {e}")




from datetime import datetime, timedelta, time
from pytz import timezone




# Assuming these functions are defined elsewhere
# def check_and_delete_trade_json(bucket, file):
#     # Returns trade data or None
# def start_buy(knockout, result):
#     # Executes buy routine

def main(test=False):
    global url
    # Set up New York timezone (EST/EDT)
    ny_timezone = timezone('America/New_York')
    
    if test:
        print("Test mode ON: No waiting, executing trades immediately.")
        trade_data = check_and_delete_trade_json(BUCKET_NAME, TRADE_FILE)
        if trade_data:
            print("Trade data found, triggering buy routine.")
            position = trade_data.get('position', [])
            if position and position[0] in ['Long', 'Short']:
                result = 1 if position[0] == 'Long' else -1
                knockout = position[1]
                start_buy(knockout, result)

                print("Position bought.")
            else:
                print("Invalid position in trade.json. Skipping buy routine.")
        else:
            print("No trade.json found in test mode. Exiting.")
        return  # Bail out after one run in test mode
    
    # Normal mode with scheduling
    current_time = datetime.now(ny_timezone)
    today = current_time.date()
    
    # Set initial buy window for today
    start_time = ny_timezone.localize(datetime.combine(today, time(10, 55)))
    end_time = ny_timezone.localize(datetime.combine(today, time(11, 30)))
    
    # If past today's buy window, move to tomorrow
    if current_time > start_time:
        start_time += timedelta(days=1)
        end_time += timedelta(days=1)
    
    # Initialize sell execution tracking
    sell_executed = True
    position_bought = False
    last_day = current_time.date()
    
    print(f"Starting main loop. Buy window: {start_time} to {end_time}, Sell trigger at 15:40 daily.")
    
    while True:
        current_time = datetime.now(ny_timezone)
        
        # Check if the day has changed
        if current_time.date() != last_day:
            sell_executed = False
            last_day = current_time.date()
            # Update buy window for the new day
            start_time = ny_timezone.localize(datetime.combine(current_time.date(), time(10, 55)))
            end_time = ny_timezone.localize(datetime.combine(current_time.date(), time(11, 30)))
        
        # Calculate sell time for today
        sell_time = ny_timezone.localize(datetime.combine(current_time.date(), time(15, 38)))
        
        # Trigger sell routine at 15:53
        if current_time >= sell_time and not sell_executed:
            print("Triggering sell routine at 15:38.")
            try:
                sell_option(url)  # Use global url set by start_buy
            except Exception as e:
                print(f"Error in sell routine: {e}")
                log_error_to_s3(str(e), BUCKET_NAME)
            sell_executed = True
            position_bought = False
        
        # Buy routine within window
        if start_time <= current_time < end_time:
            if not position_bought:
                trade_data = check_and_delete_trade_json(BUCKET_NAME, TRADE_FILE)
            if trade_data:
                print("Trade data found, triggering buy routine.")
                position = trade_data.get('position', [])
                if position and position[0] in ['Long', 'Short']:
                    result = 1 if position[0] == 'Long' else -1
                    knockout = position[1]
                    try:
                        start_buy(knockout, result)
                        sell_executed = False
                        position_bought = True
                        print(url)
                    except Exception as e:
                        print(f"Error in buy routine: {e}")
                        log_error_to_s3(str(e), BUCKET_NAME)
                else:
                    print("Invalid position in trade.json. Skipping buy routine.")
            elif not position_bought:
                print("No trade.json found. Waiting 10 seconds...")
                tm.sleep(10)
        else:
            # Wait until buy window or move to next day
            wait_time = (start_time - current_time).total_seconds()
            if wait_time > 0:
                 
                tm.sleep(min(wait_time, 10))  # Cap wait to 10 seconds
            else:
                # Move to next day's buy window
                start_time += timedelta(days=1)
                end_time += timedelta(days=1)
                print(f"Moving to next day. Next buy window at {start_time}.")
        
        # Prevent tight looping
        tm.sleep(30)

if __name__ == "__main__":
    test_mode = len(sys.argv) > 1 and sys.argv[1].lower() == 'test'
    main(test=test_mode)