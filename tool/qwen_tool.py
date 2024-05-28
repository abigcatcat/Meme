from http import HTTPStatus
import dashscope



from dashscope import MultiModalConversation

def qwen_generate(file_path1,prompt):
    """Sample of use local file.
       linux&mac file schema: file:///home/images/test.png
       windows file schema: file://D:/images/abc.png
    """
    dashscope.api_key = 'sk-885a64117b8045479513ef3c9df0c0ac'
    local_file_path1 = f'file://{file_path1}'

    messages = [{
        'role': 'system',
        'content': [{
            'text': 'You are a helpful assistant.'
        }]
    }, {
        'role':
        'user',
        'content': [

            {
                'image': local_file_path1
            },
            {
                'text': 'Based on this image of a meme, answer the following question:'+prompt
            },
        ]
    }]
    response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)

    return response['output']['choices'][0]['message']['content'][0]['text']


if __name__ == '__main__':
    qwen_generate('/public/home/jiac/jiac/agents/autogen_agentchat.png','What is in the image?')