from typing import Dict


class ButtonHandler:
    def __init__(self):
        """
        Accepts button states and converts them into just_pressed/just_released events.

        Useful when you want to convert immeadiate button states into events.

        Example:
            >>> button_handler = ButtonHandler()
            >>> current_button_state = {"button1": 0.0, "button2": 1.0, "trigger_value": 0.1234}
            >>> button_handler.update_buttons(current_button_state)
            >>> assert button_handler.just_pressed("button1") is False
            >>> assert button_handler.just_pressed("button2") is True
            >>> assert button_handler.get_value("trigger_value") == 0.1234
            >>> current_button_state = {"button1": 0.0, "button2": 1.0, "trigger_value": 0.5678}
            >>> button_handler.update_buttons(current_button_state)
            >>> assert button_handler.just_pressed("button1") is False
            >>> assert button_handler.just_pressed("button2") is False
            >>> assert button_handler.get_value("trigger_value") == 0.5678
        """
        self._button_states = {}
        self._prev_states = {}

    def update_button(self, button_name: str, current_value: float) -> None:
        """
        Updates the state of a button with its current value

        Args:
            button_name: (str) Name of the button
            current_value: (float) Current value of the button
        """
        if button_name in self._button_states:
            self._prev_states[button_name] = self._button_states[button_name]
        else:
            self._prev_states[button_name] = 0.0
        self._button_states[button_name] = current_value

    def update_buttons(self, button_states: Dict[str, float]) -> None:
        """
        Updates the state of multiple buttons with their current values

        Args:
            button_states: (Dict[str, float]) Dictionary of button names and their current values
        """
        for button_name, current_value in button_states.items():
            self.update_button(button_name, current_value)

    def just_pressed(self, button_name: str, threshold: float = 0.6) -> bool:
        """
        Returns True if button was just pressed (transition from below to above threshold)

        Args:
            button_name: (str) Name of the button
            threshold: (float) Threshold value for considering button pressed (0.0-1.0)

        Returns:
            (bool) True if button was just pressed, False otherwise
        """
        if button_name not in self._button_states:
            return False
        return (self._prev_states[button_name] < threshold and
                self._button_states[button_name] >= threshold)

    def just_released(self, button_name: str, threshold: float = 0.4) -> bool:
        """
        Returns True if button was just released (transition from above to below threshold)

        Args:
            button_name: (str) Name of the button
            threshold: (float) Threshold value for considering button released (0.0-1.0)

        Returns:
            (bool) True if button was just released, False otherwise
        """
        if button_name not in self._button_states:
            return False
        return (self._prev_states[button_name] >= threshold and
                self._button_states[button_name] < threshold)

    def get_value(self, button_name: str) -> float:
        """
        Gets the current value of a button

        Args:
            button_name: (str) Name of the button

        Returns:
            (float) Current value of the button
        """
        assert button_name in self._button_states, f"Button {button_name} was not updated"

        return self._button_states[button_name]

    def is_pressed(self, button_name: str, threshold: float = 0.6) -> bool:
        """
        Returns True if button is currently pressed (value >= threshold)

        Args:
            button_name: (str) Name of the button
            threshold: (float) Threshold value for considering button pressed (0.0-1.0)

        Returns:
            (bool) True if button is currently pressed, False otherwise
        """
        if button_name not in self._button_states:
            return False
        return self._button_states[button_name] >= threshold

    def is_released(self, button_name: str, threshold: float = 0.4) -> bool:
        """
        Returns True if button is currently released (value < threshold)

        Args:
            button_name: (str) Name of the button
            threshold: (float) Threshold value for considering button released (0.0-1.0)

        Returns:
            (bool) True if button is currently released, False otherwise
        """
        if button_name not in self._button_states:
            return False
        return self._button_states[button_name] < threshold

    @property
    def has_updates(self) -> bool:
        """
        Returns True if any button has been updated since the last update
        """
        return any(self._prev_states.values() != self._button_states.values())
