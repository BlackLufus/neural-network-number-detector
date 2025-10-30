from enum import Enum

import numpy as np
import pygame
import os
import cv2

from neural_network_numpy import NeuralNetwork

class DrawCanvas:

    user_input = '1'
    text_output = ''
    nn = None

    def __init__(self, nn_model_path = None, width:int=400, height:int=400, brush_size:int=20, save_size:int=28, output_folder:str='output', input_layer_size=784, hidden_layer_size=[128, 64], output_layer_size=10):
        self.WIDTH = width
        self.HEIGHT = height
        self.BRUSH_SIZE = brush_size
        self.SAVE_SIZE = save_size
        self.OUTPUT_FOLDER = output_folder

        # Init and load NN model
        if nn_model_path:
            self.nn = NeuralNetwork(input_layer_size, hidden_layer_size, output_layer_size)
            self.nn.load(nn_model_path)

    def __show_text(self, text):
        self.text_output = str(text)
        print(text)

    def __save_image(self, label):
        output_path = f"{self.OUTPUT_FOLDER}/{label}"
        os.makedirs(output_path, exist_ok=True)
        filename = f"{len(os.listdir(output_path)):04d}.png"
        filepath = os.path.join(output_path, filename) if label is not None else "temp/temp.png"

        # Screenshot vom Canvas speichern
        pygame.image.save(self.SCREEN, filepath)

        # In Graustufen + Größe 50x50 umwandeln
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = img[60:460, ]
        img = cv2.resize(img, (self.SAVE_SIZE, self.SAVE_SIZE), dst=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
        print(filepath)
        cv2.imwrite(filepath, img)
        print(f"✅ Saved as {filepath}")
        return filepath

    def build_and_run(self):
        # Init pygame
        pygame.init()
        self.SCREEN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        base_font = pygame.font.Font(None, 64)
        pygame.display.set_caption("Ziffern-Zeichner (drücke 's' zum Speichern, 'c' zum Löschen)")

        # Init rect for value selection
        input_rect = pygame.Rect(0, 0, 400, 60)
        color = pygame.Color('chartreuse4') if not self.nn else pygame.Color(25, 25, 25)

        # Colors for drawing
        WHITE = (255, 255, 255) # Background
        BLACK = (0, 0, 0)       # color
        self.SCREEN.fill(WHITE)

        # Set frames per second
        self.clock = pygame.time.Clock()
        self.clock.tick(120)

        # init varibales
        is_ctrl_pressed = False
        is_input_active = False
        is_drawing = False

        # make output folder if not exist yet
        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)

        # Main loop
        is_running = True # is running
        while is_running:
            for event in pygame.event.get():

                # Stop loop if quit was released
                if event.type == pygame.QUIT:
                    is_running = False

                # Check for mouse button down in pygame
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if input_rect.collidepoint(event.pos):
                        is_input_active = True
                    else:
                        is_input_active = False
                        is_drawing = True

                # Check for mouse button up
                elif event.type == pygame.MOUSEBUTTONUP:
                    print("Mouseup")
                    is_drawing = False
                
                    if self.nn:
                        filepath = self.__save_image(None)
                        print(filepath)
                        label, probs = self.nn.predict(filepath)
                        print(probs)
                        self.__show_text(f"OUTPUT: {label} ({probs[label]:.5f})")

                elif event.type == pygame.KEYDOWN:
                    if is_input_active and not self.nn:
                        if event.unicode.isdigit():
                            self.user_input = event.unicode
                            self.__show_text(f"INPUT: {event.unicode}")
                        elif event.key == pygame.K_MINUS:
                            self.user_input = '-'
                            self.__show_text(f"INPUT: -")
                    
                    # CTRL pressed
                    if event.key == pygame.K_LCTRL:
                        is_ctrl_pressed = True

                    # store image if ctrl + s is pressed
                    if is_ctrl_pressed and event.key == pygame.K_s:
                        self.__save_image(self.user_input)
                        self.SCREEN.fill(WHITE)

                    # clear canvas if ctrl + s is pressed
                    elif is_ctrl_pressed and event.key == pygame.K_c:
                        self.SCREEN.fill(WHITE)
                
                elif event.type == pygame.KEYUP:
                    # release ctrl
                    if event.key == pygame.K_LCTRL:
                        is_ctrl_pressed = False

            # Wenn Maus gedrückt ist → zeichnen
            if is_drawing:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                pygame.draw.circle(self.SCREEN, BLACK, (mouse_x, mouse_y), self.BRUSH_SIZE)

            pygame.draw.rect(self.SCREEN, color, pygame.Rect(0, 0, 400, 60))
            text_surface = base_font.render(self.text_output, True, (255, 255, 255))
            self.SCREEN.blit(text_surface, (input_rect.x, input_rect.y))


            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    # dc = DrawCanvas(output_folder="data/train")
    # dc = DrawCanvas("models/nn_number_detector_tiny_01234_0001.npz")
    dc = DrawCanvas(
        "models/nn_number_detector_numpy_0004.npz",
        input_layer_size=784,
        hidden_layer_size=[256, 256],
        output_layer_size=10
    )
    dc.build_and_run()