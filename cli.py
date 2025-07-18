import os
from pyfiglet import figlet_format
from rich.console import Console
from rich.panel import Panel
import questionary

console = Console()


def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")


def show_header():
    ascii_banner = figlet_format("MLP Trainer", font="small")
    console.print(ascii_banner, style="bold green")

    description = (
        "[bold white]Welcome to your CLI for training and visualizing your custom MLP model.[/bold white]\n"
        "[green]Use the menu to train, evaluate, and plot predictions from your network.[/green]"
    )

    console.print(Panel(description, expand=False, border_style="blue"))


def main_menu():
    while True:
        choice = questionary.select(
            "Choose an option:",
            choices=["Train model", "Show loss curves", "Show predictions", "Exit"],
        ).ask()

        if choice == "Exit":
            console.print("[bold red]Goodbye![/bold red]")
            break
        elif choice == "Train model":
            console.print("[bold yellow]-> You chose to train the model.[/bold yellow]")
        elif choice == "Show loss curves":
            console.print("[bold cyan]-> You chose to view loss curves.[/bold cyan]")
        elif choice == "Show predictions":
            console.print("[bold magenta]-> You chose to show predictions.[/bold magenta]")


def main():
    clear_terminal()
    show_header()
    main_menu()


if __name__ == "__main__":
    main()
