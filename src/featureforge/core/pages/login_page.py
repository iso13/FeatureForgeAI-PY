"""Login page object. TS equivalent: ``src/core/pages/loginPage.ts``.

Uses role/label locators rather than CSS so it survives markup changes; the URL
comes from the environment base URL instead of a hard-coded host.
"""

from __future__ import annotations

from featureforge.core.pages.base_page import BasePage


class LoginPage(BasePage):
    path = "login"

    def login(self, email: str, password: str) -> None:
        self.page.get_by_label("Email").fill(email)
        self.page.get_by_label("Password").fill(password)
        self.page.get_by_role("button", name="Login").click()

    def request_password_reset(self, email: str) -> None:
        self.page.get_by_text("Forgot password").click()
        self.page.get_by_label("Email").fill(email)
        self.page.get_by_role("button", name="Reset password").click()

    def change_password(self, new_password: str) -> None:
        self.page.get_by_label("New password").fill(new_password)
        self.page.get_by_label("Confirm password").fill(new_password)
        self.page.get_by_role("button", name="Change password").click()

    @property
    def confirmation_message(self) -> str:
        return str(self.page.locator(".confirmation-message").inner_text())

    def is_logged_in(self) -> bool:
        return bool(self.page.get_by_text("Welcome").is_visible()) or "/dashboard" in self.page.url
