"""
vast.ai 셋업 스크립트 - 한 번만 실행하면 됨
사용법: python setup_et_models.py --et2-checkpoint ./checkpoints/et_predictor2_seed123

ET model 1 (Huang & Hollenstein 2023):
  - SelectiveCacheForLM 레포에서 가중치 받아서 eyetrackpy 패키지 내부에 복사

ET model 2 (Li & Rudzicz 2021):
  - 사용자가 학습한 체크포인트(.pt / .safetensors) 경로를 환경변수로 등록
"""

import argparse
import os
import shutil
import subprocess
import sys


def run(cmd, check=True):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=check)


def find_eyetrackpy_root():
    try:
        import eyetrackpy
        return os.path.dirname(eyetrackpy.__file__)
    except ImportError:
        return None


def install_packages():
    print("\n[1/4] eyetrackpy, tokenizer_aligner 설치 중...")
    run("pip install git+https://github.com/angelalopezcardona/tokenizer_aligner.git@v1.0.0 -q")
    run("pip install git+https://github.com/angelalopezcardona/eyetrackpy.git@v1.0.0 -q")
    run("pip install safetensors -q")


def setup_et_model1(clone_dir="./SelectiveCacheForLM"):
    print("\n[2/4] ET model 1 가중치 설정 중 (Huang & Hollenstein 2023)...")

    # SelectiveCacheForLM 클론
    if not os.path.isdir(clone_dir):
        run(f"git clone https://github.com/huangxt39/SelectiveCacheForLM.git {clone_dir}")
    else:
        print(f"  {clone_dir} 이미 존재, 스킵")

    src = os.path.join(clone_dir, "FPmodels", "T5-tokenizer-BiLSTM-TRT-12-concat-3")
    if not os.path.isfile(src):
        raise FileNotFoundError(
            f"가중치 파일을 찾을 수 없습니다: {src}\n"
            "SelectiveCacheForLM 레포 구조가 바뀌었을 수 있습니다. 직접 확인해주세요."
        )

    # eyetrackpy 패키지 내부 경로에 복사
    et_root = find_eyetrackpy_root()
    if et_root is None:
        raise ImportError("eyetrackpy가 설치되지 않았습니다. install_packages()를 먼저 실행하세요.")

    dst_dir = os.path.join(
        et_root, "data_generator", "fixations_predictor_trained_1"
    )
    dst = os.path.join(dst_dir, "T5-tokenizer-BiLSTM-TRT-12-concat-3")
    os.makedirs(dst_dir, exist_ok=True)

    if not os.path.isfile(dst):
        shutil.copy2(src, dst)
        print(f"  가중치 복사 완료: {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")
    else:
        print(f"  이미 존재: {dst}")


def setup_et_model2(checkpoint_path):
    print("\n[3/4] ET model 2 체크포인트 확인 중 (Li & Rudzicz 2021)...")

    # .pt 또는 .safetensors 중 존재하는 것 탐색
    resolved = None
    for ext in ["", ".safetensors", ".pt"]:
        candidate = checkpoint_path + ext if not checkpoint_path.endswith(ext) else checkpoint_path
        if os.path.isfile(candidate):
            resolved = candidate
            break

    if resolved is None:
        raise FileNotFoundError(
            f"ET model 2 체크포인트를 찾을 수 없습니다: {checkpoint_path}[.pt/.safetensors]\n"
            "노트북에서 학습한 checkpoints/et_predictor2_seed123.pt (또는 .safetensors)를 "
            "지정해주세요."
        )

    print(f"  체크포인트 확인: {resolved} ({os.path.getsize(resolved)/1e6:.1f} MB)")

    # 환경변수 등록 안내 + .env 파일 생성
    abs_path = os.path.abspath(resolved)
    env_line = f"ET2_CHECKPOINT_PATH={abs_path}"

    with open(".env_et", "w") as f:
        f.write(env_line + "\n")

    print(f"\n  아래 명령을 실행하거나 .env_et를 source 해주세요:")
    print(f"  export {env_line}")
    print(f"  또는: source .env_et")

    # 현재 프로세스에도 등록
    os.environ["ET2_CHECKPOINT_PATH"] = abs_path
    return abs_path


def verify_setup():
    print("\n[4/4] 설치 검증 중...")

    # eyetrackpy model 1
    try:
        from eyetrackpy.data_generator.fixations_predictor_trained_1.fixations_predictor_model_1 import FixationsPredictor_1
        print("  ✓ FixationsPredictor_1 import OK")
    except Exception as e:
        print(f"  ✗ FixationsPredictor_1 import 실패: {e}")

    # et2_wrapper
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from et2_wrapper import FixationsPredictor_2
        print("  ✓ FixationsPredictor_2 (wrapper) import OK")
    except Exception as e:
        print(f"  ✗ FixationsPredictor_2 wrapper import 실패: {e}")
        print("    et2_wrapper.py가 같은 디렉토리에 있는지 확인하세요.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--et2-checkpoint",
        default="./checkpoints/et_predictor2_seed123",
        help="ET model 2 체크포인트 경로 (확장자 없어도 됨, .pt/.safetensors 자동 탐색)",
    )
    parser.add_argument(
        "--skip-install", action="store_true",
        help="pip install 생략 (이미 설치된 경우)"
    )
    parser.add_argument(
        "--clone-dir", default="./SelectiveCacheForLM",
        help="SelectiveCacheForLM 클론 경로"
    )
    args = parser.parse_args()

    if not args.skip_install:
        install_packages()

    setup_et_model1(args.clone_dir)
    setup_et_model2(args.et2_checkpoint)
    verify_setup()

    print("\n✓ 셋업 완료. 이제 train을 실행하세요.")
    print("  예시: python rlhf_rw/main.py -d OpenAssistant/oasst1 -m meta-llama/Meta-Llama-3-8B \\")
    print("          --concat True --use_softprompt True -fmv 2 --features_used 0,1,0,1,0 --seed 44")


if __name__ == "__main__":
    main()
