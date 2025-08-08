from fastapi import Header, HTTPException

API_KEY = {"test": 5}

def verify_api_key(x_api_key:str=Header(None)):
    credits = API_KEY.get(x_api_key, 0)
    if not credits:
        raise HTTPException(status_code=401, detail="Invalid api key or insufficent credits")
    return x_api_key
