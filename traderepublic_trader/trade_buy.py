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
import time
from datetime import datetime
from datetime import timedelta
from pytz import timezone
from apscheduler.schedulers.blocking import BlockingScheduler
import math
import re

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
            time.sleep(10)
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
def login():
    phone, code = get_secret(SECRET_NAME, REGION_NAME)
    if not phone or not code:
        return None

    options = Options()
    options.add_argument("start-maximized")
    options.add_argument("--window-size=1920,1080")

    driver = uc.Chrome(options=options, use_subprocess=False)
    driver.get("https://app.traderepublic.com/login")
    driver.implicitly_wait(5)
    time.sleep(5)

    # Accept cookies
    try:
        button = driver.find_element(By.XPATH, "//button[@class='buttonBase consentCard__action buttonPrimary consentCard__action']//span[text()='Accept All']")
        button.click()
    except Exception as e:
        print(f"Warning: Could not accept cookies: {e}")

    # Enter phone number
    input_element = driver.find_element(By.ID, 'loginPhoneNumber__input')
    for char in phone:
        input_element.send_keys(char)
        time.sleep(0.5)
    input_element.send_keys(Keys.RETURN)
    time.sleep(1)

    # Enter initial code
    actions = ActionChains(driver)
    for char in code:
        actions.send_keys(char).perform()
        time.sleep(0.5)
    actions.send_keys(Keys.RETURN).perform()
    time.sleep(1)

    # Delete login.txt from S3 before requesting 2FA code
    delete_s3_file(BUCKET_NAME, 'login.txt')

    # Wait for and read the new 2FA code from S3
    two_factor_code = read_s3_file(BUCKET_NAME, 'login.txt')
    if not two_factor_code:
        print("Failed to retrieve 2FA code. Exiting login.")
        driver.quit()
        return None

    # Enter the 2FA code
    actions.send_keys(two_factor_code + Keys.RETURN).perform()
    time.sleep(5)

    # Handle "Got it" button if it appears
    try:
        ok_button = driver.find_element(By.XPATH, "//span[@class='buttonBase__title' and text()='Got it']")
        ok_button.click()
    except Exception:
        pass

    return driver

# Existing trading functions (unchanged, just included for completeness)
def find_nearest_knockout(searched_knockout, direction, driver):
    rows = driver.find_elements(By.XPATH, "//tbody[@class='browseDerivativesLayout__tableBody']/tr[not(@aria-hidden='true')]")
    selected_row = None
    for i, row in enumerate(rows):
        if "future" in row.text.lower():
            continue
        parts = row.text.split('\n')
        try:
            value_str = parts[2].replace(' Pkt.', '').replace(' $', '').replace(' €', '').replace(' Pts.', '').replace(',', '')
            value = float(value_str)
            if (value - searched_knockout) * direction < -10:
                row.click()
                selected_row = i
                WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//span[@role="status"]')))
                break
        except Exception as e:
            print(f"Error processing row: {row.text} - {str(e)}")
            continue
    if selected_row is not None:
        price_element = driver.find_element(By.XPATH, '//span[@role="status"]')
        price_value = float(price_element.text.replace('€', '').strip())
        if price_value < 2.0:
            driver.back()
            WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, "//tbody[@class='browseDerivativesLayout__tableBody']/tr[not(@aria-hidden='true')]")))
            rows = driver.find_elements(By.XPATH, "//tbody[@class='browseDerivativesLayout__tableBody']/tr[not(@aria-hidden='true')]")
            for j in range(selected_row + 1, len(rows)):
                row = rows[j]
                if "future" in row.text.lower():
                    continue
                try:
                    row.click()
                    WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//span[@role="status"]')))
                    price_value = float(driver.find_element(By.XPATH, '//span[@role="status"]').text.replace('€', '').strip())
                    if price_value >= 2.0:
                        return True
                except Exception:
                    continue
            return False
        return True
    return False

def find_option(knockout, direction, driver):
    if direction == 1:
        url2 = "https://app.traderepublic.com/derivatives/US6311011026/knockouts?tab=long&sort=knockout:desc"
    else:
        url2 = "https://app.traderepublic.com/derivatives/US6311011026/knockouts?tab=short&sort=knockout:asc"
    driver.get(url2)
    while True:
        if find_nearest_knockout(knockout, direction, driver):
            break
        driver.execute_script("window.scrollBy(0, 800);")
        time.sleep(1)
    return driver.current_url

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
        time.sleep(0.5)
        shares_value = driver.execute_script("return arguments[0].value;", shares_input)
        price_value = driver.execute_script("return arguments[0].value;", price_input)
        if shares_value != str(num_shares) or float(price_value) != limit_price:
            print(f"Warning: Values not set correctly. Shares: {shares_value}, Price: {price_value}")
            
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
        WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CLASS_NAME, 'dropdownList__listbox')))
        actions = ActionChains(driver)
        actions.send_keys(Keys.DOWN).send_keys(Keys.DOWN).send_keys(Keys.ENTER).perform()
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//li[@id="orderBegin__modeSelection-stopMarket" and contains(@class, "-selected")]')))
        print("Successfully switched to Stop Order")
        
    except Exception as e:
        print(f"Error switching to Stop Order: {e}")
        

def set_stop_order_details(driver):
    try:
        WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CLASS_NAME, 'orderFlowCard')))
        shares_available_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'orderFormAvailableFunds')))
        num_shares = int(re.search(r'\d+', shares_available_element.find_element(By.TAG_NAME, 'p').text.strip()).group())
        buy_in_element = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME, 'instrumentPosition__buyIn')))
        buy_in_price = float(buy_in_element.text.strip().replace('€', '').replace(',', '.'))
        stop_price = round(buy_in_price * 0.76, 2)
        shares_input = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'orderFlow__stopMarketsellShares')))
        stop_price_input = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, 'orderFlow__stopMarketsellPrice')))
        driver.execute_script("arguments[0].click();", shares_input)
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, 'orderFlow__stopMarketsellShares')))
        actions = ActionChains(driver)
        actions.move_to_element(shares_input).click().send_keys(str(num_shares)).perform()
        driver.execute_script("arguments[0].click();", stop_price_input)
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, 'orderFlow__stopMarketsellPrice')))
        actions.move_to_element(stop_price_input).click().send_keys(str(stop_price)).perform()
        time.sleep(0.5)
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
        

def get_current_url(driver):
    try:
        current_url = driver.current_url
        print(f"Current URL: {current_url}")
        return current_url
    except Exception as e:
        print(f"Error getting current URL: {e}")
        return None

def start_buy(knockout, direction):
    driver = login()
    time.sleep(5)
    url = find_option(knockout, direction, driver)
    
    time.sleep(5)
    switch_to_limit_order(driver)
      
    time.sleep(2)
    set_order_details(driver)
       
    time.sleep(2)
    review_order(driver)
    
    time.sleep(2)
    try:
        click_and_press_down(driver)
    except Exception as e:
        print(f"Warning: Failed to press Down keys: {e}")
    time.sleep(2)
    buy_now(driver)
        
    time.sleep(2)
    reload_page(driver)
    time.sleep(2)
    click_sell(driver)
    time.sleep(2)
    switch_to_stop_order(driver) 
        
    time.sleep(2)
    set_stop_order_details(driver)
        
    time.sleep(2)
    review_order(driver)
        
    time.sleep(2)
    try:
        click_and_press_down(driver)
    except Exception as e:
        print(f"Warning: Failed to press Down keys: {e}")
    time.sleep(2)
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

def main():
    # Set up scheduler for New York time (EST)
    sched = BlockingScheduler(timezone=timezone('America/New_York'))
    start_time = datetime.now(timezone('America/New_York')).replace(hour=10, minute=55, second=0, microsecond=0)
    end_time = datetime.now(timezone('America/New_York')).replace(hour=11, minute=30, second=0, microsecond=0)

    if start_time < datetime.now(timezone('America/New_York')) < end_time:
        start_time = start_time + timedelta(days=1)

    def job():
        current_time = datetime.now(timezone('America/New_York'))
        if current_time >= end_time:
            print("Time exceeded 11:30 AM EST. Terminating scheduler.")
            sched.shutdown()
            return

        trade_data = check_and_delete_trade_json(BUCKET_NAME, TRADE_FILE)
        if trade_data:
            print("Trade data found, triggering buy routine.")
            result = 1 if trade_data.get('position')[0] == 'Long' else -1 if trade_data.get('position')[0] == 'Short' else None
            if result is not None:
                knockout = trade_data.get('position')[1]
                start_buy(knockout, result)
            else:
                print("Invalid position in trade.json. Skipping buy routine.")
        else:
            print("No trade.json found. Waiting 10 seconds...")
            time.sleep(10)

    # Schedule the job to run at 10:55 AM EST
    sched.add_job(job, 'cron', hour=10, minute=55, timezone=timezone('America/New_York'))

    try:
        print(f"Scheduler started. Next run at 10:55 AM EST, terminating at 11:30 AM EST.")
        sched.start()
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in scheduler: {e}")
        log_error_to_s3(error_traceback, BUCKET_NAME, LOG_FILE)

if __name__ == "__main__":
    main()