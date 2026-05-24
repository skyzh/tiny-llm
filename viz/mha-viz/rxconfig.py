import reflex as rx

config = rx.Config(
    app_name="mha_viz",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ]
)