import io
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

try:
    import gradio as gr
except Exception:
    gr = None

try:
    # normal package import when running as module
    from .model import SimplePoseNet, AdvancedPoseNet, MultiAgentPoseNet
except Exception:
    # fallback when running the file directly: add project root to sys.path and import by absolute name
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pose_train.model import SimplePoseNet, AdvancedPoseNet, MultiAgentPoseNet


def _to_euler(qw, qx, qy, qz):
    # Convert quaternion (w,x,y,z) to Euler angles (roll, pitch, yaw) in degrees
    # Reference: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    # roll (x-axis rotation)
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(t3, t4)

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def preprocess_pil(img: Image.Image, use_imagenet_norm: bool, img_size=(240, 320)):
    # returns torch tensor (1, C, H, W)
    if use_imagenet_norm:
        img = img.convert('RGB').resize((img_size[1], img_size[0]))
        arr = np.array(img).astype(np.float32) / 255.0
        # H,W,C -> C,H,W
        arr = arr.transpose(2, 0, 1)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        arr = (arr - mean) / std
    else:
        img = img.convert('L').resize((img_size[1], img_size[0]))
        arr = np.array(img).astype(np.float32) / 255.0
        # add channel dim
        arr = np.expand_dims(arr, 0)

    t = torch.from_numpy(arr).unsqueeze(0).float()
    return t


@lru_cache(maxsize=8)
@lru_cache(maxsize=16)
def get_model(model_type: str, backbone: str, pretrained: bool, use_imagenet_norm: bool, max_agents: int = 3):
    """Return a model instance for the given configuration (cached)."""
    in_ch = 3 if use_imagenet_norm else 1
    model_type = (model_type or 'simple')
    if model_type == 'simple':
        model = SimplePoseNet(in_channels=in_ch, backbone=backbone, pretrained=pretrained)
    elif model_type == 'advanced':
        model = AdvancedPoseNet(in_channels=in_ch, backbone=backbone, pretrained=pretrained, use_attention=True)
    elif model_type == 'multi_agent':
        model = MultiAgentPoseNet(in_channels=in_ch, backbone=backbone, pretrained=pretrained, max_agents=max_agents, use_attention=True)
    else:
        model = SimplePoseNet(in_channels=in_ch, backbone=backbone, pretrained=pretrained)

    model.eval()
    return model


def predict(model_type, backbone, pretrained, use_imagenet_norm, pil_img: Image.Image, max_agents: int = 3):
    if gr is None:
        return 'Gradio is not installed. Install requirements and try again.', None
    model = get_model(model_type, backbone, bool(pretrained), bool(use_imagenet_norm), int(max_agents))
    x = preprocess_pil(pil_img, use_imagenet_norm)
    with torch.no_grad():
        out = model(x)

    # If the model returns (poses, presence) -> multi-agent
    if isinstance(out, (tuple, list)) and len(out) == 2:
        poses, presence = out
        poses = poses.squeeze(0).cpu().numpy()  # [max_agents, 7]
        presence = presence.squeeze(0).cpu().numpy()
        lines = []
        for i in range(min(len(poses), int(max_agents))):
            if presence[i] > 0.5:
                p = poses[i]
                tx, ty, tz = p[:3].tolist()
                qw, qx, qy, qz = p[3:].tolist()
                roll, pitch, yaw = _to_euler(qw, qx, qy, qz)
                lines.append(f"Agent{i}: t=({tx:.3f},{ty:.3f},{tz:.3f}) euler=({roll:.1f},{pitch:.1f},{yaw:.1f})")
        text = "\n".join(lines) if lines else "No agents detected"
        img = pil_img.convert('RGB').resize((320, 240))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.rectangle([(0, 0), (320, max(24, 16 * len(lines)))], fill=(0, 0, 0, 127))
        for i, line in enumerate(text.split('\n')):
            draw.text((6, 6 + 16 * i), line, fill=(255, 255, 255), font=font)
        return text, img

    # Otherwise single-agent output
    out_np = out.squeeze(0).cpu().numpy()
    tx, ty, tz = out_np[:3].tolist()
    qw, qx, qy, qz = out_np[3:].tolist()
    roll, pitch, yaw = _to_euler(qw, qx, qy, qz)

    text = f"Translation (m): x={tx:.3f}, y={ty:.3f}, z={tz:.3f}\n"
    text += f"Rotation (quat): w={qw:.4f}, x={qx:.4f}, y={qy:.4f}, z={qz:.4f}\n"
    text += f"Euler (deg): roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}"

    img = pil_img.convert('RGB').resize((320, 240))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle([(0, 0), (320, 60)], fill=(0, 0, 0, 127))
    draw.text((6, 6), text.split('\n')[0], fill=(255, 255, 255), font=font)
    draw.text((6, 22), text.split('\n')[1], fill=(255, 255, 255), font=font)
    draw.text((6, 38), text.split('\n')[2], fill=(255, 255, 255), font=font)

    return text, img


def build_ui():
    if gr is None:
        raise RuntimeError('Gradio not installed; install with `pip install gradio`')

    with gr.Blocks() as demo:
        gr.Markdown('# Vision-based Pose Estimator')
        with gr.Tab("Inference"):
            with gr.Row():
                model_type = gr.Radio(['simple', 'advanced', 'multi_agent'], label='Model Type', value='simple')
                backbone = gr.Radio(['small', 'resnet18', 'resnet50'], label='Backbone', value='small')
                pretrained = gr.Checkbox(label='Use ImageNet pretrained weights', value=False)
                use_imagenet = gr.Checkbox(label='Use ImageNet normalization (RGB)', value=False)
                max_agents = gr.Slider(minimum=1, maximum=8, step=1, value=3, label='Max agents (multi-agent only)')
            inp = gr.Image(type='pil', label='Upload image (single frame)')
            out_text = gr.Textbox(label='Predicted Pose', lines=6)
            out_img = gr.Image(label='Annotated Image')
            btn = gr.Button('Run')
            btn.click(fn=predict, inputs=[model_type, backbone, pretrained, use_imagenet, inp, max_agents], outputs=[out_text, out_img])

        with gr.Tab("Metrics"):
            gr.Markdown('Aggregated metrics across runs in `work/run_*`: one plot for train loss, one for val loss.')
            with gr.Row():
                refresh_btn = gr.Button('Refresh plots')
            with gr.Row():
                train_plot = gr.Image(label='Train Loss (all runs)')
                val_plot = gr.Image(label='Val Loss (all runs)')

            def _aggregate_plots():
                from pathlib import Path
                import pandas as pd
                import matplotlib.pyplot as plt
                import io
                from PIL import Image as PILImage

                root = Path(__file__).resolve().parent.parent
                work = root / 'work'
                runs = []
                if work.exists():
                    for sub in sorted([p for p in work.iterdir() if p.is_dir()]):
                        # include only runs that look like final experiments and have results
                        if not sub.name.startswith('run_'):
                            continue
                        csv = sub / 'results.csv'
                        if csv.exists():
                            try:
                                df = pd.read_csv(csv)
                                # map run folder to a concise model label
                                name_lower = sub.name.lower()
                                if 'multi' in name_lower:
                                    label = 'MultiAgent'
                                elif 'advanced' in name_lower:
                                    label = 'Advanced'
                                elif 'simple' in name_lower:
                                    label = 'Simple'
                                else:
                                    label = sub.name
                                runs.append((label, df))
                            except Exception:
                                pass

                # helper to create a plot and return PIL Image
                def _make_plot(kind: str):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    any_plotted = False
                    for name, df in runs:
                        if kind == 'train' and 'train_loss' in df.columns:
                            ax.plot(df['epoch'], df['train_loss'], label=name)
                            any_plotted = True
                        elif kind == 'val' and 'val_loss' in df.columns:
                            ax.plot(df['epoch'], df['val_loss'], label=name)
                            any_plotted = True
                    ax.set_xlabel('epoch')
                    ax.set_ylabel('loss')
                    ax.set_title(f'{kind.capitalize()} loss across runs')
                    if any_plotted:
                        ax.legend()
                    else:
                        ax.text(0.5, 0.5, f'No {kind} data found', ha='center', va='center')
                    fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, bbox_inches='tight')
                    buf.seek(0)
                    plt.close(fig)
                    return PILImage.open(buf)

                train_img = _make_plot('train')
                val_img = _make_plot('val')
                return train_img, val_img

            refresh_btn.click(fn=_aggregate_plots, inputs=[], outputs=[train_plot, val_plot])
            demo.load(fn=_aggregate_plots, inputs=[], outputs=[train_plot, val_plot])

    return demo


if __name__ == '__main__':
    if gr is None:
        print('Gradio not installed. Install with pip install gradio')
    else:
        demo = build_ui()
        try:
            demo.launch(server_name='127.0.0.1', server_port=7860, share=False)
        except OSError:
            demo.launch(server_name='127.0.0.1', server_port=None, share=False)
