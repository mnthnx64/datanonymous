from selenium import webdriver
import numpy as np
import cv2
import dlib
import imutils

class FaceSwapper():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    @staticmethod
    def get_image_from_internet():
        driver = webdriver.Safari()
        driver.set_window_size(800,600)
        driver.get('https://thispersondoesnotexist.com/')
        # driver.find_element_by_xpath('//*[@id="face"]')
        driver.save_screenshot('images/Face.png')
        driver.close()

    def load_images(self):
        self.img = cv2.imread("images/Face.png")
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        self.img2 = cv2.imread("images/test1.jpeg")
        self.img2_gray = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        height, width, channels = self.img2.shape
        self.img2_new_face = np.zeros((height, width, channels), np.uint8)

    @staticmethod
    def extract_index_nparray(nparray):
        index = None
        for num in nparray[0]:
            index = num
            break
        return index

    def find_faces(self):
        mask = np.zeros_like(self.img_gray)
        faces = self.detector(self.img_gray)
        for face in faces:
            landmarks = self.predictor(self.img_gray, face)
            landmarks_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))

            points = np.array(landmarks_points, np.int32)
            convexhull = cv2.convexHull(points)
            # cv2.polylines(self.img, [convexhull], True, (255, 0, 0), 3)
            cv2.fillConvexPoly(mask, convexhull, 255)

            face_image_1 = cv2.bitwise_and(self.img, self.img, mask=mask)

            rect = cv2.boundingRect(convexhull)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(landmarks_points)
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)

            indexes_triangles = []
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])


                index_pt1 = np.where((points == pt1).all(axis=1))
                index_pt1 = self.extract_index_nparray(index_pt1)

                index_pt2 = np.where((points == pt2).all(axis=1))
                index_pt2 = self.extract_index_nparray(index_pt2)

                index_pt3 = np.where((points == pt3).all(axis=1))
                index_pt3 = self.extract_index_nparray(index_pt3)

                if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                    triangle = [index_pt1, index_pt2, index_pt3]
                    indexes_triangles.append(triangle)

        faces2 = self.detector(self.img2_gray)
        for face in faces2:
            landmarks = self.predictor(self.img2_gray, face)
            landmarks_points2 = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points2.append((x, y))


            points2 = np.array(landmarks_points2, np.int32)
            convexhull2 = cv2.convexHull(points2)

        lines_space_mask = np.zeros_like(self.img_gray)
        lines_space_new_face = np.zeros_like(self.img2)

        for triangle_index in indexes_triangles:
            # Triangulation of the first face
            tr1_pt1 = landmarks_points[triangle_index[0]]
            tr1_pt2 = landmarks_points[triangle_index[1]]
            tr1_pt3 = landmarks_points[triangle_index[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)


            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1
            cropped_triangle = self.img[y: y + h, x: x + w]
            cropped_tr1_mask = np.zeros((h, w), np.uint8)


            points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                            [tr1_pt2[0] - x, tr1_pt2[1] - y],
                            [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

            # Lines space
            cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
            cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
            cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
            lines_space = cv2.bitwise_and(self.img, self.img, mask=lines_space_mask)

            # Triangulation of second face
            tr2_pt1 = landmarks_points2[triangle_index[0]]
            tr2_pt2 = landmarks_points2[triangle_index[1]]
            tr2_pt3 = landmarks_points2[triangle_index[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


            rect2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect2

            cropped_tr2_mask = np.zeros((h, w), np.uint8)

            points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

            # Warp triangles
            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

            # Reconstructing destination face
            img2_new_face_rect_area = self.img2_new_face[y: y + h, x: x + w]
            img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

            img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
            self.img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area



        # Face swapped (putting 1st face into 2nd face)
        img2_face_mask = np.zeros_like(self.img2_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)

        img2_head_noface = cv2.bitwise_and(self.img2, self.img2, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, self.img2_new_face)

        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

        seamlessclone = cv2.seamlessClone(result, self.img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

        cv2.imshow("Steganographed", imutils.resize(seamlessclone,height=500))
        cv2.imwrite('outputs/swapped.png', seamlessclone)
        
        grf = np.bitwise_and(cv2.cvtColor(img2_head_mask,cv2.COLOR_GRAY2BGR),self.img2)
        gr = cv2.cvtColor(grf,cv2.COLOR_BGR2GRAY)
        row = []
        col =[]
        for idx,i in enumerate(gr):
            if np.max(i) != np.min(i): row.append(idx)

        frt = np.transpose(gr)
        for idx2, i in enumerate(frt):
            if np.max(i) != np.min(i): col.append(idx2)
        srow = sorted(row)
        scol = sorted(col)
        cutted = self.img2[srow[0]:srow[-1],scol[0]:scol[-1]]
        cv2.imshow('Original face', imutils.resize(cutted,height=200))
        cv2.imshow('Decoded Face', imutils.resize(self.img2, height=500))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    fs = FaceSwapper()
    fs.get_image_from_internet()
    fs.load_images()
    fs.find_faces()

