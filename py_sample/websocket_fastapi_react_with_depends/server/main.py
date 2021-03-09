from typing import Optional

from fastapi import Cookie, Depends, FastAPI, Query, WebSocket, status
from fastapi.responses import HTMLResponse

app = FastAPI()


with open('./index.html', 'rt') as f:
    html = f.read()

@app.get("/")
async def get():
    return HTMLResponse(html)

async def get_cookie_or_token(
    websocket: WebSocket,
    session: Optional[str] = Cookie(None),
    token: Optional[str] = Query(None)
):
    if session is None and token is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
    return session or token


@app.websocket("/items/{item_id}/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    item_id: str,
    q: Optional[int] = None,
    cookie_or_token: str = Depends(get_cookie_or_token)
):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(
            f"Session cookie or query token value is: {cookie_or_token}"
        )
        if q is not None:
            await websocket.send_text(f"Query parameter q is: {q}")
        await websocket.send_text(f"Message text was: {data}, for item ID: {item_id}")
