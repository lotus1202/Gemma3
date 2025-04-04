import sys
sys.path.insert(0, r'E:\mypackages')

import os
import argparse
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import shutil

def convert_onnx_to_tflite(onnx_path, tflite_path):
    # 一時的に TensorFlow の SavedModel としてエクスポートするためのディレクトリ
    saved_model_dir = "temp_saved_model"

    print("Loading ONNX model from:", onnx_path)
    # ONNX モデルを読み込む
    onnx_model = onnx.load(onnx_path)
    
    print("Converting ONNX model to TensorFlow SavedModel...")
    # ONNX モデルから TensorFlow モデルへの変換
    tf_rep = prepare(onnx_model)
    # SavedModel としてエクスポートする
    tf_rep.export_graph(saved_model_dir)
    
    print("Converting TensorFlow SavedModel to TFLite model...")
    # TensorFlow SavedModel から TFLite コンバーターでモデルを変換する
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    print("Writing TFLite model to:", tflite_path)
    # 変換された TFLite モデルを指定のパスに保存する
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    # 一時ディレクトリの削除
    shutil.rmtree(saved_model_dir)
    print("Conversion to TFLite completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert an ONNX model of Gemma3 to TensorFlow Lite format")
    parser.add_argument('--input', type=str, required=True,
                        help="入力 ONNX モデルファイルのパス（例: gemma3.onnx）")
    parser.add_argument('--output', type=str, required=True,
                        help="出力 TFLite モデルファイルのパス（例: gemma3.tflite）")
    args = parser.parse_args()
    convert_onnx_to_tflite(args.input, args.output)
