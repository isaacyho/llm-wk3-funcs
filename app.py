from dotenv import load_dotenv
import chainlit as cl
import movie_functions

load_dotenv()

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
 
client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\
You are a personal assistant.  If the user requests a list of currently playing movies in the theater,
print the following line and only this line:
XXX nowplaying

If the user requests show times for a particular movie, print the following line and only this line:
XXX showtimes <moviename> <zipcode>

If a single user request combines multiple functions, print each function as a separate line in your response.

Otherwise, respond normally.
"""

@observe
@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)

@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)
    
    await response_message.update()

    return response_message

@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    
    print("generating response...")
    while True:
        response_message = await generate_response(client, message_history, gen_kwargs)

        # Split the response content by newline
        response_lines = response_message.content.split('\n')
        
        # Process each line separately
        for line in response_lines:
            if line.startswith("XXX"):
                # call specialized function
                # split by whitespace
                parts = line.split()
                for part in parts:
                    print("Part: {part}")
                if parts[1] == "nowplaying":
                    print("Calling the function to get the list of currently playing movies")
                    movies = movie_functions.get_now_playing_movies()
                    message_history.append({"role": "system", "content": f"movies: {movies}"})
                    response_message.content += f"Movie: {movies}"
                elif parts[1] == "showtimes" and len(parts) == 4:
                    movie_name = parts[2]
                    zipcode = parts[3]
                    print(f"Calling the function to get showtimes for {movie_name} in {zipcode}")
                    showtimes = movie_functions.get_showtimes(movie_name, zipcode)
                    print(f"showtimes: {showtimes}")
                    message_history.append({"role": "system", "content": f"showtimes: {showtimes}"})
                    response_message.content += f"{showtimes}"
                else:
                    print("Unknown command")
                
                response_message = await generate_response(client, message_history, gen_kwargs)
            else:
                response_message.content = line
                break  # Use the first non-XXX line as the response

        else:
            break

    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.main()
