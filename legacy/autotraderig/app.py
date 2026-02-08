import time
import random
import json
import boto3
import logging
import traceback
from math import floor
import os
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
#html = driver.find_element_by_tag_name('html')
from decimal import Decimal
from io import BytesIO
from tempfile import mkdtemp
import random
import uuid

buying_options = [
    "Strapping in for a wild ride! Firing up these options! 🎢",
    "Full steam ahead! Adding more fuel to our rocket! 🚀",
    "Get ready for liftoff! These options are about to skyrocket! 🌠",
    "Locked and loaded! Picking up these options like hot cakes! 🥞",
    "This ain't no casino, but we're betting big on these options! 🎲",
    "Time to place our bets! These options have the Midas touch. 💰",
    "Throwing our hat in the ring! Let's see where these options take us. 🎩",
    "Ready to rumble! These options have got us fired up! 🔥",
    "Who needs a lottery ticket when we've got these options? 🎟️",
    "Buckle up! We're hitting the stonk highway! 🛣️",
    "Time to put the pedal to the metal with these options! 🏎️",
    "Hold on tight! We're diving into the option ocean! 🌊",
    "Let's get this party started with these fresh options! 🎉",
    "Revving up the engine for these high-octane options! 🏁",
    "Options, assemble! Time to make some serious moves! 🦸",
    "Jumping into the fray with these power-packed options! 💪",
    "Ready, set, go! We're sprinting toward option glory! 🏃",
    "Setting sail on the stonk sea with these shiny options! ⛵",
    "We're on the hunt for treasure, and these options are our map! 🗺️"
]

selling_options_wins = [
    
    "One small step for man, one giant leap for our profits! 🌌",
    "The eagle has landed! Sold for gains, and we're feeling fly! 🦅",
    "Cha-ching! Another successful mission for our rocket crew! 🚀",
    "Just sold for a profit, and now we're dancing to the tendie tango! 💃",
    "They say money talks, and our gains are singing sweet melodies! 🎶",
    "That's a wrap! Cashing in on our winnings like a boss. 😎",
    "Out with a bang! These gains are lighting up the sky. 🎆",
    "Just secured a first-class ticket on the Profit Express! 🚆💰",
    "Cashing in those chips! 🎰 Just sold those options and it feels like hitting the jackpot without even setting foot in Vegas. Riding high on the profit express! 🚂💰🎉",
    "Touchdown! We're making gains like a pro athlete! 🏈",
    "Scored a slam dunk with these gains! 🏀",
    "Sold for profit, and now we're soaring sky-high! 🦅",
    "We've hit the jackpot with these gains! 🎰",
    "Cha-ching! Sold for a sweet, sweet profit! 💰",
    "Another day, another dollar! Cashing in on these gains. 💵",
    "Gains for days! We're on a winning streak! 🏆",
    "We've struck gold with these gains! Time to celebrate! 🎉",
    "Profits in the bag! Sold like a true WallStreetBets champ! 🥊",
    "Just made some serious bank! Cashing out like a king! 👑"
]

selling_options_losses = [
    "Took a hit, but we're still swinging for the fences! ⚾",
    "No pain, no gain! We'll rise like a phoenix from the ashes. 🔥",
    "Bumpy landing, but we're ready to refuel and take off again! 🛬🚀",
    "Took a dip in the red sea, but we're ready to sail toward green waters! 🌊",
    "They say every cloud has a silver lining, and we're hunting for ours! ☁️",
    "Sold for a loss, but we're still in the game! Time for a comeback! 💪",
    "We might be down, but we're not out! Time to regroup and reload! 🔄",
    "Even WallStreetBets legends need a little hiccup to keep things interesting! 😉",
    "We're shaking off this loss and getting back on the rocket! 🚀",
     "A little setback, but we're gearing up for the next round! 🥊",
    "Shaking off the dust and getting ready for the next sprint! 🏃",
    "We may have stumbled, but we're picking ourselves up and moving forward! 💪",
    "Sometimes you win, sometimes you learn. We'll get 'em next time! 🎯",
    "Chin up! We're learning from this loss and coming back stronger! 💪",
    "Every pro was once a beginner. We'll bounce back from this loss! 🦘",
    "We fell short this time, but we're just getting started! 🚀",
    "Even the greatest stumble sometimes. Time to regroup and conquer! ⚔️",
    "It's not about how many times you fall, but how many times you get back up! 🆙",
    "Sold for a loss, but we're still smiling and ready for the next challenge! 😁"
]
stop_loss_messages = [
    "Ouch, that stop loss stings, but we'll bounce back even stronger! 💪",
    "Hit the stop loss, but we're not down for the count! We'll rise again! 🦸",
    "Stop loss triggered, but we're just refueling for the next big launch! 🚀",
    "We've hit a speed bump, but the road to success is still ahead! 🛣️",
    "A stop loss today means we're gearing up for a stronger tomorrow! 🌅",
    "Took a hard hit with that stop loss, but we're still in the game! 🎮",
    "We may have hit a stop loss, but our determination is unstoppable! ⚡",
    "Even the best hit a stop loss sometimes. Time to recalibrate and refocus! 🔍",
    "Sold at a stop loss, but we're ready to turn the page and start a new chapter! 📖",
    "The stop loss is just a detour on our path to victory! Let's keep pushing forward! 🏁",
    "Brace for impact! 🌠 We hit the stop loss, but that's just a pitstop on our racetrack to the stars. Tighten those seatbelts, because the next lap might just be our victory lap! 🏁🚀🌟"
]



s3 = boto3.resource("s3")
os.environ['SELENIUM_SERVER_EXECUTABLE'] = '/usr/bin/chromedriver'
# Initialize the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS services clients
secrets_manager_client = boto3.client("secretsmanager", region_name="eu-west-1")
sns_client = boto3.client("sns", region_name="eu-west-1")

# SNS topic ARN
sns_topic_arn = "arn:aws:sns:eu-west-1:REDACTED_AWS_ACCOUNT_ID:ig-messages"

table_name = 'turnarount'
primary_key_name = 'val'
primary_key_value = 'turnaroundday'

def put_value_in_dynamodb(table_name, primary_key_name, primary_key_value, value):
    # Create a DynamoDB client
    dynamodb = boto3.resource('dynamodb')
    decimal_value = Decimal(str(value))
    # Get the table
    table = dynamodb.Table(table_name)

    # Create the item to put in the table
    item = {
        primary_key_name: primary_key_value,
        'value': decimal_value
    }

    # Put the item in the table
    response = table.put_item(Item=item)

    return response


def take_screenshot(driver):
   

    screenshot = driver.get_screenshot_as_png()
    
    s3.Object('REDACTED_BUCKET', f'screenshots/{str(uuid.uuid4())}.png').put(Body=screenshot)

    return screenshot



def get_secret():
    secret_name = "ig.com"
    response = secrets_manager_client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])

secrets = get_secret()
email = secrets["email"]
password = secrets["password"]

# Helper function to publish SNS messages
def publish_sns_message(message, subject="IG Message"):
    try:
        sns_client.publish(
            TopicArn=sns_topic_arn,
            Message=json.dumps(message, ensure_ascii=False),
            Subject=subject
        )
    except Exception as e:
        logger.error(f"Failed to publish SNS message: {str(e)}")

def warmup():
    options = webdriver.ChromeOptions()
    service = Service(executable_path=r'/opt/chromedriver')
    options.binary_location = '/opt/chrome/chrome'
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--start-maximized")
    options.add_argument("--single-process")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-dev-tools")
    options.add_argument("--no-zygote")
    options.add_argument(f"--user-data-dir={mkdtemp()}")
    options.add_argument(f"--data-path={mkdtemp()}")
    options.add_argument(f"--disk-cache-dir={mkdtemp()}")
    options.add_argument("--remote-debugging-port=9222")

    # Use the correct executable path for ChromeDriver
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_window_size(1920, 1080)
    url = "https://google.com"
    driver.maximize_window()
    driver.get(url)
    time.sleep(60)

def login(account = "Turbo24"):
    options = webdriver.ChromeOptions()
    service = Service(executable_path=r'/opt/chromedriver')
    options.binary_location = '/opt/chrome/chrome'
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--start-maximized")
    options.add_argument("--single-process")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-dev-tools")
    options.add_argument("--no-zygote")
    options.add_argument(f"--user-data-dir={mkdtemp()}")
    options.add_argument(f"--data-path={mkdtemp()}")
    options.add_argument(f"--disk-cache-dir={mkdtemp()}")
    options.add_argument("--remote-debugging-port=9222")

    # Use the correct executable path for ChromeDriver
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_window_size(1920, 1080)
    url = "https://www.ig.com/de/login"
    driver.maximize_window()
    driver.get(url)
    # Wait for the page to load
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "account_id")))
    
    # Check for cookie preference and accept if present
    try:
        cookie_accept_button = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "onetrust-accept-btn-handler")))
        cookie_accept_button.click()
    except:
        logger.info("Cookie preference widget not found.")
    email_field = driver.find_element(By.NAME,"account_id")
    email_field.send_keys(email)
    password_field = driver.find_element(By.ID,"nonEncryptedPassword")
    password_field.send_keys(password)
    time.sleep(5)
    login_button = driver.find_element(By.ID,"loginbutton")
    login_button.click()
    time.sleep(5)
    wait = WebDriverWait(driver, 10)

    max_retry = 5
    for i in range(max_retry):
        try:
            account_names = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".name-value a")))
            break
        except:
            time.sleep(1)
            if i < max_retry - 1:  # i is zero indexed
                continue
            else:
                raise
    turbo24_account = None
    for account_name in account_names:
        if account_name.text.strip() == account:
            turbo24_account = account_name
            account_row = account_name.find_element(By.XPATH,"./ancestor::div[contains(@class, 'accounts-table-row')]")
            break

    if turbo24_account:
        print(f" account found")

        balance = account_row.find_element(By.CSS_SELECTOR,".funds-column span").text.strip()
        profit_loss = account_row.find_element(By.CSS_SELECTOR,".profit-loss-column span").text.strip()
        platform_open_btn = account_row.find_element(By.CSS_SELECTOR,".segmented-button-label")

        print(f"Account Balance: {balance}")
        print(f"Account Profit/Loss: {profit_loss}")

        # Click on "Platform öffnen" button using JavaScript
        driver.execute_script("arguments[0].click();", platform_open_btn)
    else:
        print("Turbo24 account not found")
    return driver
def scroll_element_into_view(driver, element):
    driver.execute_script("arguments[0].scrollIntoView();", element)


def findposition(position, driver):
    try:
        long_position = driver.find_element(By.CSS_SELECTOR, "li[data-automation='CALL']")
        short_position = driver.find_element(By.CSS_SELECTOR, "li[data-automation='PUT']")
    except:
        time.sleep(1)
        long_position = driver.find_element(By.CSS_SELECTOR, "li[data-automation='CALL']")
        short_position = driver.find_element(By.CSS_SELECTOR, "li[data-automation='PUT']")
    
    if position[0] == "Long":
        long_position.click()
    else:
        short_position.click()
    
    time.sleep(1)
    rows = driver.find_elements(By.CSS_SELECTOR, "li.ig-grid_row")
    desired_knockout = position[1]
    
    # Extract all knockout levels and sort them
    knockout_levels = sorted([float(row.find_element(By.CSS_SELECTOR, ".ticket-deal-now-mtf_strike-level").text) for row in rows])
    
    # Print all available knockout levels for logging purposes
    for level in knockout_levels:
        print(f"Available knockout level: {level}")
    
    if position[0] == "Long":
        # Find the highest knockout level that is below the desired_knockout
        suitable_levels = [level for level in knockout_levels if level < desired_knockout]
        selected_knockout = suitable_levels[-1] if suitable_levels else knockout_levels[0]
    
    else:  # Short position
        # Find the lowest knockout level that is above the desired_knockout
        suitable_levels = [level for level in knockout_levels if level > desired_knockout]
        selected_knockout = suitable_levels[0] if suitable_levels else knockout_levels[-1]
    
    # Print the selected knockout level for logging purposes
    print(f"Selected knockout level: {selected_knockout}")
    
    # Find the row corresponding to the selected_knockout
    for row in rows:
        if float(row.find_element(By.CSS_SELECTOR, ".ticket-deal-now-mtf_strike-level").text) == selected_knockout:
            time.sleep(1)
            row.click()
            break

    
def limit_order(driver):
    #take_screenshot(driver)
    dropdown_element = WebDriverWait(driver, 60).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, ".basic-dropdown_trigger"))
)
    scroll_element_into_view(driver, dropdown_element)
    #take_screenshot(driver)
    #s3.Object('REDACTED_BUCKET', f'html/beforeclick').put(Body=html_source)
    # Click the dropdown to open it
    dropdown_element.click()
    #s3.Object('REDACTED_BUCKET', f'html/afterclick').put(Body=html_source)
    #take_screenshot(driver)
    time.sleep(1)
    # Wait for the dropdown menu to open and find the "Limit" option
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Wait for the dropdown menu to open and find the "Limit" option
            limit_option = WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "li[data-value='LIMIT']"))
            )

            # Wait for the "Limit" option to become clickable
            WebDriverWait(driver, 60).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "li[data-value='LIMIT']"))
            )
            time.sleep(1)
            
            # Click the "Limit" option
            limit_option.click()
            time.sleep(1)
            
            break  # If the click was successful, break out of the loop
        except selenium.common.exceptions.StaleElementReferenceException:
            if attempt < max_retries - 1:  # i.e. if it's not the last attempt
                continue  # try again
            else:
                raise  
    # Find the available capital
    capital_element = WebDriverWait(driver, 60).until(
    EC.presence_of_element_located((By.CSS_SELECTOR,'.account-balances_figure'))
)
    scroll_element_into_view(driver, capital_element)

    
    capital_text = capital_element.text.replace('.', '').replace(',', '.').replace('\xa0€', '')
    take_screenshot(driver)
    available_capital = float(capital_text.replace("€",""))

    # Find the price of the option
    price_element = driver.find_element(By.CSS_SELECTOR,'.price-tick--dealable')
    price = float(price_element.text)

    # Calculate the new price and update the price input field
    new_price = price * 1.05
    price_input = input_field = driver.find_element(By.CSS_SELECTOR,'.numeric-input-with-incrementors_input')
    price_input.clear()
    price_input.send_keys(f"{new_price:.2f}")

    # Calculate the initial number of options to buy
    fees = 3
    capital_to_invest = available_capital - fees
    num_options = floor(capital_to_invest / new_price)

    while True:
        # Update the number of options input field
        options_input = driver.find_element(By.CSS_SELECTOR,'.numeric-input-with-incrementors_input.text-right')
        options_input.clear()
        options_input.send_keys(str(num_options))
    
        # Check the projected cost
        projected_cost_element = driver.find_element(By.CSS_SELECTOR,'.ticket-deal-now-mtf_consideration-value')
        projected_cost_text = projected_cost_element.text.replace('.', '').replace(',', '.').replace('\xa0€', '')
        projected_cost = float(projected_cost_text.replace("€",""))

        # Adjust the number of options if the projected cost is too high
        if projected_cost > capital_to_invest:
            num_options -= 1
        else:
            break
    # Click the 'Order platzieren' button
    submit_button = driver.find_element(By.CSS_SELECTOR,"button[data-automation='ig-action-submit-button']")
    time.sleep(1)
    submit_button.click()
    
def number_of_orders_and_position(driver):
    try:
        open_positions_element = driver.find_element(By.CSS_SELECTOR,".platform-navigation_menu-item--positions .badge-with-number")
    except:
        time.sleep(1)
        open_positions_element = driver.find_element(By.CSS_SELECTOR,".platform-navigation_menu-item--positions .badge-with-number")
    num_open_positions = int(open_positions_element.text.strip())

# Find the number of open orders
    try:
        open_orders_element = driver.find_element(By.CSS_SELECTOR,".platform-navigation_menu-item--orders .badge-with-number")
    except:
        time.sleep(1)
        open_orders_element = driver.find_element(By.CSS_SELECTOR,".platform-navigation_menu-item--orders .badge-with-number")
    num_open_orders = int(open_orders_element.text.strip())
    return num_open_positions, num_open_orders
def find_open_position(driver):
    open_positions_button = driver.find_element(By.CSS_SELECTOR,"[data-automation='openPositionsFlyout']")
    time.sleep(1)
    scroll_element_into_view(driver, open_positions_button)
    take_screenshot(driver)
    #open_positions_button.click()
    driver.execute_script("arguments[0].click();", open_positions_button)
    time.sleep(1)
    open_position = driver.find_element(By.CSS_SELECTOR,".ig-grid_row")

    # Extract the amount of options
    take_screenshot(driver)
    scroll_element_into_view(driver, open_position.find_element(By.CSS_SELECTOR,".cell-size"))
    amount_of_options = int(open_position.find_element(By.CSS_SELECTOR,".cell-size").text.strip().replace("+", ""))

    # Extract the type
    position_type = open_position.find_element(By.CSS_SELECTOR,".cell-market-name_name").text.strip()

    # Extract the current Gewinn/Verlust
    gewinn_verlust = open_position.find_element(By.CSS_SELECTOR,".cell-profit-loss").text.strip()

    # Extract and store the Eröffnung price value
    eroeffnung_price = float(open_position.find_element(By.CSS_SELECTOR,".cell-open-level").text.strip())

    elements = {"position": position_type, "size": amount_of_options, "entry": eroeffnung_price, "win/loss": gewinn_verlust}

    return eroeffnung_price,amount_of_options, position_type, gewinn_verlust, elements

def find_open_position_observation(driver):
    open_positions_button = driver.find_element(By.CSS_SELECTOR,"[data-automation='openPositionsFlyout']")
    time.sleep(1)
    scroll_element_into_view(driver, open_positions_button)

    #open_positions_button.click()
    driver.execute_script("arguments[0].click();", open_positions_button)
    time.sleep(1)
    try:
        open_position = driver.find_element(By.CSS_SELECTOR,".ig-grid_row")
    except:
        open_positions_button = driver.find_element(By.CSS_SELECTOR,"[data-automation='openPositionsFlyout']")
        time.sleep(1)
        scroll_element_into_view(driver, open_positions_button)

        open_positions_button.click()
        time.sleep(1)
        open_position = driver.find_element(By.CSS_SELECTOR,".ig-grid_row")
        

    # Extract the amount of options
    scroll_element_into_view(driver, open_position.find_element(By.CSS_SELECTOR,".cell-size"))
    amount_of_options = int(open_position.find_element(By.CSS_SELECTOR,".cell-size").text.strip().replace("+", ""))

    # Extract the type
    position_type = open_position.find_element(By.CSS_SELECTOR,".cell-market-name_name").text.strip()

    # Extract the current Gewinn/Verlust
    gewinn_verlust = open_position.find_element(By.CSS_SELECTOR,".cell-profit-loss").text.strip()

    # Extract and store the Eröffnung price value
    eroeffnung_price = float(open_position.find_element(By.CSS_SELECTOR,".cell-open-level").text.strip())

    return {"position": position_type, "size": amount_of_options, "entry": eroeffnung_price, "win/loss": gewinn_verlust}

     

def set_stop_order(driver):
    position =find_open_position(driver)
    open_position = driver.find_element(By.CSS_SELECTOR,".ig-grid_row")
    # Click on sell button
    sell_button = open_position.find_element(By.CSS_SELECTOR,".cell-close_btn")
    scroll_element_into_view(driver, sell_button)
    time.sleep(1)

    sell_button.click()
    time.sleep(1)
    dropdown_element = WebDriverWait(driver, 60).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, ".basic-dropdown_trigger"))
)
    time.sleep(1)
    scroll_element_into_view(driver, dropdown_element)
# Click the dropdown to open it
    dropdown_element.click()

# Wait for the dropdown menu to open and find the "Limit" option
    stop_option = WebDriverWait(driver, 60).until(
    EC.presence_of_element_located((By.XPATH, "//li[contains(@data-value, 'STOP_MARKET')]"))
    )

# Wait for the "Stop" option to become clickable
    WebDriverWait(driver, 60).until(
    EC.element_to_be_clickable((By.XPATH, "//li[contains(@data-value, 'STOP_MARKET')]"))
    )

# Click the "Stop" option
    stop_option.click()
    time.sleep(1)
    stop_level = round(position[0] * 0.71, 3)

# Find the Stop-Level input field and set its value
    stop_level_input = WebDriverWait(driver, 60).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, ".trigger-level_input input"))
)
    stop_level_input.clear()
    stop_level_input.send_keys(str(stop_level))
    order_button = WebDriverWait(driver, 60).until(
    EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-automation='ig-action-submit-button']"))
)
    order_button.click()
def delete_order(driver):
    orders_button = WebDriverWait(driver, 60).until(
    EC.element_to_be_clickable((By.CSS_SELECTOR, "li[data-automation='openOrdersFlyout']"))
)
    scroll_element_into_view(driver, orders_button)
    orders_button.click()
    time.sleep(3)
    take_screenshot(driver)
    driver.execute_script("window.scrollTo(0, 0);")
    driver.execute_script("document.body.style.zoom='80%'")

    take_screenshot(driver)
    '''
    cancel_order_button = WebDriverWait(driver, 60).until(
    EC.element_to_be_clickable((By.CSS_SELECTOR, "button.delete"))
)
    scroll_element_into_view(driver, cancel_order_button)
    cancel_order_button.click()
    '''
    time.sleep(1)
    button = driver.find_element(By.CSS_SELECTOR,'.btn.btn-grid.btn-secondary.delete')
    driver.execute_script("arguments[0].click();", button)
def sell_market(driver):
    position =find_open_position(driver)
    time.sleep(1)
    open_position = driver.find_element(By.CSS_SELECTOR,".ig-grid_row")
    # Click on sell button
    sell_button = open_position.find_element(By.CSS_SELECTOR,".cell-close_btn")
    scroll_element_into_view(driver, sell_button)
    #sell_button.click()
    driver.execute_script("arguments[0].click();", sell_button)
    time.sleep(1)
    dropdown_element = WebDriverWait(driver, 60).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, ".basic-dropdown_trigger"))
)
    scroll_element_into_view(driver, dropdown_element)
    
    driver.execute_script("arguments[0].click();", dropdown_element)
    time.sleep(1)
    take_screenshot(driver)
    option = WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.XPATH, "//li[contains(@data-value, 'MARKET')]")))
    time.sleep(1)
    driver.execute_script("arguments[0].click();", option)
    take_screenshot(driver)
    time.sleep(1) 
    order_button = WebDriverWait(driver, 60).until( EC.presence_of_element_located((By.CSS_SELECTOR, "button[data-automation='ig-action-submit-button']")))
    driver.execute_script("arguments[0].click();", order_button)
def buy(position,account = "Turbo24"):
    try:
        driver = login(account=account)
    except:
        time.sleep(5)
        driver = login(account=account)
    time.sleep(5)
    findposition(position,driver)
    time.sleep(5)
    limit_order(driver)
    time.sleep(5)
    set_stop_order(driver)
    time.sleep(2)
    open_position_info =find_open_position_observation(driver)
    print(open_position_info)
    driver.close()
    return open_position_info
def sell(account = "Turbo24"):
    try:
        driver = login(account=account)
    except:
        time.sleep(5)
        driver = login(account=account)
    time.sleep(5)
   
    number_of_orders, number_of_positions = number_of_orders_and_position(driver)
    if number_of_orders == 1 and number_of_positions== 1:
        time.sleep(5)
        open_position_info = find_open_position_observation(driver)
        time.sleep(5)
        delete_order(driver)
        time.sleep(5)
        sell_market(driver)
        driver.close()
        open_position_info["exist"] = True
        return open_position_info
    else:
        return {"exist": False} 
    
def calculate_percentage(position_info):
    buy = position_info["entry"] *   position_info["size"]
    win_loss =   float(position_info["win/loss"].replace("€","").replace(".","").replace(",",".").strip())
    percentage = win_loss / buy * 100
    if percentage >= 50:
        emoji = "🚀🌓"
    elif percentage >= 30:
        emoji = "🚀🚀🚀"
    elif percentage >= 20:
        emoji = "🚀🚀"
    elif percentage >= 10:
        emoji = "🚀"
    elif percentage >= 5:
        emoji = "👍👍"
    elif percentage > 0:
        emoji = "👍"
    elif percentage == 0:
        emoji = "➖"  # No change
    elif percentage > -5:
        emoji = "🔻"
    elif percentage > -15:
        emoji = "🔻🔻"
    else:
        emoji = "📉"

    formatted_percentage = f"{percentage:.2f}% {emoji}"
    return formatted_percentage

def handler(event, context):
    try:
        if event["action"] == "BUY":
            position_info = buy(event["position"], account=event["account"])
            print("Bought position:", position_info)
                
            message = {**{"status" : random.choice(buying_options),"result": "success", "action": event["action"], "account":event["account"]}, **position_info}
            
            publish_sns_message(message)
            return {"sucess" : 1}
            
        elif event["action"] == "SELL":
            position_info = sell(account=event["account"])
            if not position_info["exist"]:
                message = {**{"status": random.choice(stop_loss_messages)},**{"result": "fail", "action": event["action"], "account":event["account"]}   }
                put_value_in_dynamodb(table_name, primary_key_name, "winloss", 0)
                publish_sns_message(message)
                return {"sucess" : 1}
            else:

                position_info["percentage win/loss"] = calculate_percentage(position_info)
                message = {**{"status" : "","result": "success", "action": event["action"], "account":event["account"]}, **position_info}
                if message["win/loss"].find("-") != -1:
                    message["status"] = random.choice(selling_options_losses)
                    put_value_in_dynamodb(table_name, primary_key_name, "winloss", 0)
                else:
                    message["status"] = random.choice(selling_options_wins)
                    put_value_in_dynamodb(table_name, primary_key_name, "winloss", 1)
                del message["exist"]


                publish_sns_message(message)
                return {"sucess" : 1}
        else:
            warmup()
      
        
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to execute action: {str(e)}")
        publish_sns_message({"result": "error", "action": event["action"], "error": str(e)}, subject="IG Error")
        return {"sucess" : 0}
