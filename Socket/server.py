import asyncio
import websockets

async def hello(websockets, path):
    name = await websockets.recv()
    print(f"< {name}")

    greeting = f"Hello {name}!"

    await websockets.send(greeting)
    print(f"> {greeting}")

start_server = websockets.serve(hello, "192.168.1.51", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()