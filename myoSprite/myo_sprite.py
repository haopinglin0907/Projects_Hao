# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:12:28 2021

@author: haopi
"""

import arcade
import random
import numpy as np

SPRITE_SCALING = 0.5

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Myo_sprite"

MOVEMENT_SPEED = 5


class Player(arcade.Sprite):

    def update(self):
        """ Move the player """
        #         # Move player.
        #         # Remove these lines if physics engine is moving player.
        #         self.center_x += self.change_x
        #         self.center_y += self.change_y

        # Check for out-of-bounds
        if self.left < 0:
            self.left = 0
        elif self.right > - 1:
            self.right = SCREEN_WIDTH - 1

        if self.bottom < 0:
            self.bottom = 0
        elif self.top > SCREEN_HEIGHT - 1:
            self.top = SCREEN_HEIGHT - 1


# --- Constants ---
SPRITE_SCALING_PLAYER = 0.5
SPRITE_SCALING_COIN = .25
COIN_COUNT = 30


class MyGame(arcade.Window):
    """ Our custom Window Class"""

    def __init__(self):
        """ Initializer """
        # Call the parent class initializer
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.Window.set_location(self, x=50, y=50)
        # Variables that will hold sprite lists
        self.player_list = None
        self.coin_list = None
        self.MOVEMENT_SPEED = 5

        # Set up the player info
        self.player_sprite = None
        self.score = 0
        self.label = 'Rest'
        self.color = arcade.color.WHITE
        self.similarity = 0

        # Track the current state of what key is pressed
        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False

        self.sound_powerup = arcade.load_sound(":resources:sounds/coin2.wav")

        arcade.set_background_color(arcade.color.AMAZON)

    def setup(self):
        """ Set up the game and initialize the variables. """

        # Sprite lists
        self.player_list = arcade.SpriteList()
        self.coin_list = arcade.SpriteList()

        # Score
        self.score = 0
        self.label = 'Rest'
        self.color = arcade.color.WHITE

        # Set up the player
        # Character image from kenney.nl

        img = ":resources:images/animated_characters/female_person/femalePerson_idle.png"
        self.player_sprite = arcade.Sprite(img, SPRITE_SCALING_PLAYER)
        self.player_sprite.center_x = 50
        self.player_sprite.center_y = 50
        self.player_list.append(self.player_sprite)

        # Create the coins
        for i in range(COIN_COUNT):
            # Create the coin instance
            # Coin image from kenney.nl
            coin = arcade.Sprite(":resources:images/items/coinGold.png",
                                 SPRITE_SCALING_COIN)

            # Position the coin
            coin.center_x = random.randrange(SCREEN_WIDTH)
            coin.center_y = random.randrange(SCREEN_HEIGHT)

            # Add the coin to the lists
            self.coin_list.append(coin)

    def on_draw(self):
        """ Draw everything """
        arcade.start_render()
        self.coin_list.draw()
        self.player_list.draw()

        # Put the score on the screen
        output = f"Score: {self.score}"
        arcade.draw_text(output, 10, 20, arcade.color.WHITE, 14)

        # Put the gesture on the screen
        arcade.draw_text(self.label, 10, SCREEN_HEIGHT - 40, arcade.color.WHITE, 20)

        # Put the similarity on the screen
        arcade.draw_text("Similarity", SCREEN_WIDTH - 120, SCREEN_HEIGHT - 40, arcade.color.WHITE, 20)
        arcade.draw_text(f"{np.abs(self.similarity):.2f}", SCREEN_WIDTH - 93, SCREEN_HEIGHT - 180, self.color, 20)

        # Draw the similarity bar on the screen
        arcade.draw_lrtb_rectangle_outline(SCREEN_WIDTH - 80, SCREEN_WIDTH - 50, SCREEN_HEIGHT - 150 + 100,
                                           SCREEN_HEIGHT - 150, arcade.color.WHITE, border_width=3)
        arcade.draw_lrtb_rectangle_filled(SCREEN_WIDTH - 78, SCREEN_WIDTH - 52,
                                          SCREEN_HEIGHT - 150 + float(self.similarity) * 100, SCREEN_HEIGHT - 150,
                                          self.color)

    # functions that also allows the control using keyboard

    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """

        if key == arcade.key.UP:
            self.up_pressed = True
        elif key == arcade.key.DOWN:
            self.down_pressed = True
        elif key == arcade.key.LEFT:
            self.left_pressed = True
        elif key == arcade.key.RIGHT:
            self.right_pressed = True

    def on_key_release(self, key, modifiers):
        """Called when the user releases a key. """

        if key == arcade.key.UP:
            self.up_pressed = False
        elif key == arcade.key.DOWN:
            self.down_pressed = False
        elif key == arcade.key.LEFT:
            self.left_pressed = False
        elif key == arcade.key.RIGHT:
            self.right_pressed = False

    def on_update(self, delta_time):

        """ Movement and game logic """
        # Calculate speed based on the keys pressed
        self.player_sprite.change_x = 0
        self.player_sprite.change_y = 0

        if self.up_pressed and not self.down_pressed:
            self.player_sprite.change_y = self.MOVEMENT_SPEED
        elif self.down_pressed and not self.up_pressed:
            self.player_sprite.change_y = -self.MOVEMENT_SPEED
        if self.left_pressed and not self.right_pressed:
            self.player_sprite.change_x = -self.MOVEMENT_SPEED
        elif self.right_pressed and not self.left_pressed:
            self.player_sprite.change_x = self.MOVEMENT_SPEED

        # Call update on all sprites (The sprites don't do much in this
        # example though.)
        self.coin_list.update()

        # Generate a list of all sprites that collided with the player.
        coins_hit_list = arcade.check_for_collision_with_list(self.player_sprite,
                                                              self.coin_list)

        # Loop through each colliding sprite, remove it, and add to the score.
        for coin in coins_hit_list:
            coin.remove_from_sprite_lists()
            # arcade.play_sound(self.sound_powerup)
            self.score += 1

        # Check for out-of-bounds
        if self.player_sprite.left < 0:
            self.player_sprite.left = 0
        elif self.player_sprite.right > SCREEN_WIDTH - 1:
            self.player_sprite.right = SCREEN_WIDTH - 1

        if self.player_sprite.bottom < 0:
            self.player_sprite.bottom = 0
        elif self.player_sprite.top > SCREEN_HEIGHT - 1:
            self.player_sprite.top = SCREEN_HEIGHT - 1

        # Call update to move the sprite
        # If using a physics engine, call update player to rely on physics engine
        # for movement, and call physics engine here.
        self.player_list.update()


def main():
    """ Main method """
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
