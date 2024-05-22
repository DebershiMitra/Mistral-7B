import boto3
import json
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def invoke_mistral_7b(client, prompt, context):
    """
    Invokes the Mistral 7B model to run an inference using the input
    provided in the request body.

    :param prompt: The prompt that you want Mistral to complete.
    :return: List of inference responses from the model.
    """

    try:
        # Mistral instruct models provide optimal results when
        # embedding the prompt into the following template:
        instruction = f"""
            <s>[INST] Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer.
            Context: {context}
            Question: {prompt}
            Answer: [/INST]
        """

        model_id = "mistral.mistral-7b-instruct-v0:2"

        body = {
            "prompt": instruction,
            "max_tokens": 200,
            "temperature": 0.5,
        }

        response = client.invoke_model(
            modelId=model_id, body=json.dumps(body)
        )

        response_body = json.loads(response["body"].read())
        outputs = response_body.get("outputs")

        completions = [output["text"] for output in outputs]

        return completions

    except ClientError:
        logger.error("Couldn't invoke Mistral 7B")
        raise
