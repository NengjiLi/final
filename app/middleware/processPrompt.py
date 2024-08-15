def promptResponse(prompt) -> str:
    instruction = (
        "You are an AI assistant helping gym's users with personalized fitness advice. "
        "Based on the user's workout summary provided, generate a concise and insightful recommendation "
        "for the user's future improvement. The advice should be clear and encouraging, no more than 50 words."
    )
    combined_prompt = f"{instruction}\n\nYour workout summary: {prompt}"

    return combined_prompt
