from nicegui import ui
from ui.chat_page import build_chat_page

ui.page("/")(build_chat_page)

ui.run(title="KMap", port=8080, reload=False, dark=True)
