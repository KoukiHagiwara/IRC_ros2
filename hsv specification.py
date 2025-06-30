#!/usr/bin/python3

#色を指定

import cv2
import numpy as np

# トラックバーのコールバック関数（何もしない）
def nothing(x):
    pass

def ball():
    # カメラを起動（0は通常PC内蔵カメラ）
    # ウェブカメラ 2
    cap = cv2.VideoCapture(2)
    cap.set(3, 640)  # 幅
    cap.set(4, 480)  # 高さ

    while True:
        ret, img = cap.read()  # フレームを取得
        if not ret:
            break
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 赤色の低い方の範囲（0〜10）
        lower_red1 = np.array([0, 58, 30])
        upper_red1 = np.array([10, 255, 255])

        # 赤色の高い方の範囲（170〜180）
        lower_red2 = np.array([170, 58, 30])
        upper_red2 = np.array([180, 255, 255])

        # 2つのマスクを作成して合成
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask1, mask2)

        # 青
        lower_blue = np.array([96, 100, 0])
        upper_blue = np.array([159, 250, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        #黄
        lower_yellow = np.array([22, 105, 81])
        upper_yellow = np.array([69, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 全色を統合
        mask = cv2.bitwise_or(mask_red, mask_blue)
        mask = cv2.bitwise_or(mask, mask_yellow)

        # 元画像とマスクを合成（該当色だけ残す）
        result = cv2.bitwise_and(img, img, mask=mask)

        # グレースケール化(これで白黒にしてる)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        #明るさ補正
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        #ノイズ除去　ガウシアン
       # blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.4)

        th1=50
        th2=100

        # Cannyエッジ検出(輪郭を検出)
        edges = cv2.Canny(blurred, th1, th2)

        #モルフォロジー
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel, iterations=2)

                # --- 円検出 ---
        circles = cv2.HoughCircles(
           # blurred,
            gradient,# 入力画像（グレースケール）
            cv2.HOUGH_GRADIENT,           # 検出手法
            dp=1,                       # 解像度の逆数（1.2が一般的）
            minDist=30,                   # 検出する円同士の最小距離
            param1=th2,                   # Cannyの上限値
            param2=35,                    # 円検出のしきい値（小さくすると検出しやすくなるが誤検出が増える）
            minRadius=25,                  # 最小円半径
            maxRadius=130                 # 最大円半径
        )

        # 円が見つかれば描画
        if circles is not None:
            circles = np.uint16(np.around(circles))  # 四捨五入して整数化
            for i in circles[0, :]:
                # 外円
                cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # 中心点
                cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)

        # 結果を表示
        cv2.imshow('Original', img)    # 元画像
        cv2.imshow('Mask', result)     # 色抽出結果
        cv2.imshow("Edge on Mask", gradient)

        # Escキー（ASCIIコード27）を押すと終了
        k = cv2.waitKey(10)
        if k == 27:
            break

    # 後処理
    cap.release()
    cv2.destroyAllWindows()

# メイン関数の呼び出し
if __name__ == "__main__":
    ball()
