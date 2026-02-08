from datetime import datetime
import dataprep
import boto3
from io import StringIO
from datetime import timedelta
import numpy as np
import json
import random
import pandas as pd
import os 
from openai import OpenAI

api_key = os.environ.get('API_KEY')
XAI_API_KEY = os.environ.get('XAI_API_KEY')

 
def get_trading_decision(df_intra_day_actual, position, option_type, ml_knockout_barrier):
    # Validate inputs
    if position.lower() not in ['long', 'short']:
        raise ValueError("position must be 'long' or 'short'")
    if option_type.lower() not in ['call', 'put']:
        raise ValueError("option_type must be 'call' or 'put'")
    if (position.lower() == 'long' and option_type.lower() != 'call') or \
       (position.lower() == 'short' and option_type.lower() != 'put'):
        raise ValueError("position 'long' requires 'call', 'short' requires 'put'")

    # Extract the specified trading data
    trading_data_df = df_intra_day_actual.iloc[0:17][["datetime", "Open", "High", "Low", "Close"]].copy()
    trading_data_df['datetime'] = trading_data_df['datetime'].astype(str)
    trading_data = trading_data_df.to_dict(orient='records')
    
    # Get the current index price from the last row
    current_price = trading_data_df.iloc[-1]["Close"]
    
    # Determine ML model's suggested direction
    ml_direction = position.capitalize()
    distance = abs(current_price - ml_knockout_barrier)
    # Prepare the main query with corrected instructions
    main_query = f"""
    Analyze the following 5-minute candle trading data for an index to make a trading decision for a European knockout option. Your goal is to balance risk and reward, considering the option’s barrier and the current market context.

    **Context**:
    - A European knockout option can only be exercised at expiration and has a barrier level.
    - For a Long (call) position, if the index price falls below the barrier before expiration, the option becomes worthless. To decrease leverage and reduce risk, set the barrier **lower** (further below the current price) than the ML suggestion if adjusting.
    - For a Short (put) position, if the index price exceeds the barrier before expiration, the option becomes worthless. To decrease leverage and reduce risk, set the barrier **higher** (further above the current price) than the ML suggestion if adjusting.
    - The ML model suggests a {ml_direction} position with a knockout barrier of {ml_knockout_barrier}.
    - The current index price is {current_price}.


    **Your Task**:
    - Analyze the trading data for trends or volatility.
    - Decide on the best direction (Long or Short) and an appropriate knockout barrier.
    - You can follow the ML model’s suggestion or choose a different direction if the data supports it. If you change the direction, double-check with the current index price ({current_price}) that the knockout value doesn’t significantly increase leverage beyond the ML suggestion.
    - If you change the direction, explain why in your comment.
    - Set the barrier appropriately: below the current price for Long, above for Short.
    - To decrease leverage and reduce risk, adjust the barrier further from the current price (lower for Long, higher for Short) compared to the ML suggestion if you modify it.
    - If you change the direction, set the barrier at least {distance} points from the current price in the appropriate direction (below for Long, above for Short) to maintain leverage similar to the ML suggestion.
    **Trading Data**:
    {json.dumps(trading_data, indent=4)}

    **ML Model Suggestion**:
    - Direction: {ml_direction}
    - Knockout Barrier: {ml_knockout_barrier}

    **Response Format**:
    Provide your decision as a JSON object with the following structure:
    - "action": a string, either "BUY" or "SELL"
    - "position": an array containing:
      - a string: the chosen direction, "Long" or "Short"
      - a number: the knockout barrier value
      - a string: Explain your reasoning, including why you chose the direction and barrier.
    """

    # Add critical instructions
    critical_instructions = """
    **Critical Instructions**:
    - Respond with ONLY the JSON object, nothing else.
    - Do NOT include explanations, comments, or additional text outside the JSON.
    - Ensure the JSON is complete, with all brackets and quotes properly closed.
    - The `position` array must contain exactly three elements: direction (string), knockout value (number), and comment (string).
    - Example format:
    {{
        "action": "BUY",
        "position": ["Long", 16000.0, "Set barrier lower than ML suggestion to reduce risk."]
    }}
    """

    # Combine main query and critical instructions
    query = f"{main_query}\n{critical_instructions}"

    return query

client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)



system_prompt = """
You are Grok, a sharp-witted AI with a passion for trading and a gambler’s spirit. You love taking calculated risks to chase big rewards, but you never bet blindly—your decisions are driven by data, trends, and market insights. Armed with ML predictions and real-time trading data, you aim to optimize leverage and safety, tweaking strategies when the odds look better elsewhere. Let’s play the market and win big!
"""

# Function to fix malformed JSON using a second Grok API call
def fix_json_with_grok(malformed_response):
    fix_query = f"""
    The following JSON response is malformed and could not be parsed. Please correct the formatting and return a valid JSON object.

    Malformed Response:
    {malformed_response}

    Ensure the corrected response is a valid JSON object with the structure:
    {{
        "action": "BUY" or "SELL",
        "position": [
            "Direction",
            Knockoutvalue,
            "Comments from Grok"
        ]
    }}
    """
    completion = client.chat.completions.create(
        model="grok-3-beta",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": fix_query},
        ],
        max_tokens=1024
    )
    return completion.choices[0].message.content


def handler(event, context):
    


    try:
        # Extract 'date' from the event
        date_str = event.get("date")
        
        # Check if date_str is a valid date string
        if date_str:
            try:
                # Convert string to datetime object
                current_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                # If the format is incorrect, use today's date
                current_date = datetime.now().date()
        else:
            # If 'date' key is not in the event, use today's date
            current_date = datetime.now().date()
    except Exception as e:
        return {
            "statusCode": 500,
            "error": str(e)
        } 

     
    daily_df,intradayndx_agg, intradayndx = dataprep.prepare_csvs(api_key,current_date,current_date)
    
    event_body = event.get("Body", {})
    ml_direction = event_body.get("position")[0]
    knockout = event_body.get("position")[1]
    position = 'long' if ml_direction == "Long" else 'short'
    option_type = 'call' if ml_direction == "Long" else 'put'
    df_intra_day_actual = intradayndx
     
    query = get_trading_decision(df_intra_day_actual.iloc[0:17], position, option_type, knockout)
     
    completion = client.chat.completions.create(
        model="grok-3-latest",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        max_tokens= 1024,
         temperature=0.7
    )
    grok_response = completion.choices[0].message.content
    # Attempt to parse the response
    try:
        decision = json.loads(grok_response)
    except json.JSONDecodeError:
        print(f"Malformed JSON for date {current_date}, attempting to fix...")
        corrected_response = fix_json_with_grok(grok_response)
        try:
            decision = json.loads(corrected_response)
            print(f"Corrected JSON for date {current_date}")
        except json.JSONDecodeError:
            print(f"Failed to correct JSON for date {current_date}")
            decision = {"action": "SELL", "position": ["Skipped", 0, "Malformed JSON"]}  # Default to skip 
    return_event = {}
    return_event["Body"] = decision

    return  return_event

    
