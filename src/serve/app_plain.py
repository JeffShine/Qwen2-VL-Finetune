import argparse
import gradio as gr
from PIL import Image
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
import warnings
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def bot_response(message, history, generation_args):
    # Initialize variables
    images = []
    videos = []

    if message["files"]:
        for file_item in message["files"]:
            if isinstance(file_item, dict):
                file_path = file_item["path"]
            else:
                file_path = file_item
            if is_video_file(file_path):
                videos.append(file_path)
            else:
                images.append(file_path)

    conversation = []
    for user_turn, assistant_turn in history:
        user_content = []
        if isinstance(user_turn, tuple):
            file_paths = user_turn[0]
            user_text = user_turn[1]
            if not isinstance(file_paths, list):
                file_paths = [file_paths]
            for file_path in file_paths:
                if is_video_file(file_path):
                    user_content.append({"type": "video", "video": file_path, "fps":1.0})
                else:
                    user_content.append({"type": "image", "image": file_path})
            if user_text:
                user_content.append({"type": "text", "text": user_text})
        else:
            user_content.append({"type": "text", "text": user_turn})
        conversation.append({"role": "user", "content": user_content})

        if assistant_turn is not None:
            assistant_content = [{"type": "text", "text": assistant_turn}]
            conversation.append({"role": "assistant", "content": assistant_content})

    user_content = []
    for image in images:
        user_content.append({"type": "image", "image": image})
    for video in videos:
        user_content.append({"type": "video", "video": video, "fps":1.0})
    user_text = message['text']
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    conversation.append({"role": "user", "content": user_content})

    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device) 

    # 设置tokenizer配置
    processor.tokenizer.clean_up_tokenization_spaces = False
    processor.tokenizer.decode_special_tokens = True
    
    # 直接生成输出
    outputs = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )
    
    # 解码输出，保留特殊token
    response = processor.tokenizer.decode(
        outputs[0], 
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
        decode_special_tokens=True
    )
    
    return response

def main(args):
    global processor, model, device

    device = args.device
    
    disable_torch_init()

    use_flash_attn = True
    
    model_name = get_model_name_from_path(args.model_path)
    
    if args.disable_flash_attention:
        use_flash_attn = False

    processor, model = load_pretrained_model(
        model_base = args.model_base, 
        model_path = args.model_path, 
        device_map=args.device, 
        model_name=model_name, 
        load_4bit=args.load_4bit, 
        load_8bit=args.load_8bit,
        device=args.device, 
        use_flash_attn=use_flash_attn
    )

    chatbot = gr.Chatbot(scale=2)
    chat_input = gr.MultimodalTextbox(
        interactive=True, 
        file_types=["image", "video"], 
        placeholder="Provide a scene graph caption of the given image.",
        show_label=False
    )
    
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }
    
    with gr.Blocks(fill_height=True) as demo:
        gr.ChatInterface(
            fn=lambda message, history: bot_response(message, history, generation_args),
            title="Qwen2.5-VL Instruct",
            stop_btn=None,  # 非流式输出不需要停止按钮
            multimodal=True,
            textbox=chat_input,
            chatbot=chatbot,
        )

    demo.queue(api_open=False)
    demo.launch(show_api=False, share=False, server_name='0.0.0.0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/jinchenhui/ustc/Qwen2-VL-Finetune/output/fft_7b_rec_conversation_42k_finetune")
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)