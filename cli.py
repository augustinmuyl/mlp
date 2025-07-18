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


def prompt_training_config():
    # Hidden layer
    layer_input = questionary.text("Hidden layers:", default="16, 32, 64").ask()
    hidden_layers = [int(x.strip()) for x in layer_input.split(",")]

    # Learning rate
    lr = questionary.text(
        "Learning rate:",
        default="0.001",
        validate=lambda val: val.replace(".", "", 1).isdigit() or "Please enter a valid number",
    ).ask()
    lr = float(lr)

    # Epochs
    epochs = questionary.text(
        "Number of training epochs:",
        default="1000",
        validate=lambda val: val.isdigit() or "Please enter a valid number",
    ).ask()
    epochs = int(epochs)
    # Note: add option for patience

    # Terminal plotting
    use_terminal_plot = questionary.confirm(
        "Use terminal plots instead of graphical ones?", default=True
    ).ask()

    return hidden_layers, lr, epochs, use_terminal_plot


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
            console.print("[bold yellow]-> Training configuration[/bold yellow]")
            hidden_layers, lr, epochs, use_terminal_plot = prompt_training_config()

            console.print(f"[white]Hidden layers:[/white] {hidden_layers}")
            console.print(f"[white]Learning rate:[/white] {lr}")
            console.print(f"[white]Epochs:[/white] {epochs}")
            console.print(f"[white]Terminal plots:[/white] {use_terminal_plot}")
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
