import os
from pyfiglet import figlet_format
from rich.console import Console
from rich.panel import Panel
import questionary
from trainer import train_model
from visualization import (
    plot_last_loss,
    plot_last_loss_terminal,
    plot_last_predictions,
    plot_last_predictions_terminal,
)

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
    if layer_input is None:
        console.print("[red]Cancelled. Returning to main menu...[/red]")
        return None, None, None
    hidden_layers = [int(x.strip()) for x in layer_input.split(",")]

    # Learning rate
    lr = questionary.text(
        "Learning rate:",
        default="0.00001",
        validate=lambda val: val.replace(".", "", 1).isdigit() or "Please enter a valid number",
    ).ask()
    if lr is None:
        console.print("[red]Cancelled. Returning to main menu...[/red]")
        return None, None, None
    if float(lr) >= 0.01:
        console.print("[yellow]⚠️ Warning: High learning rate may cause instability.[/yellow]")
    lr = float(lr)

    # Epochs
    epochs = questionary.text(
        "Number of training epochs:",
        default="1000",
        validate=lambda val: val.isdigit() or "Please enter a valid number",
    ).ask()
    if epochs is None:
        console.print("[red]Cancelled. Returning to main menu...[/red]")
        return None, None, None
    epochs = int(epochs)
    # Note: add option for patience

    return hidden_layers, lr, epochs


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

            config = prompt_training_config()
            if any(v is None for v in config):
                continue
            hidden_layers, lr, epochs = config

            console.print(f"[white]-> Starting training...[/white]")
            best_loss, best_loss_acc = train_model(hidden_layers, lr, epochs)

            console.print("\n[bold green]Training complete![/bold green]")
            console.print(f"[green]Best Loss:[/green] {best_loss:.6f}")
            console.print(f"[green]Accuracy:[/green] {best_loss_acc:.2%}")
        elif choice == "Show loss curves":
            use_terminal_plot = questionary.confirm(
                "Use terminal plots instead of graphical ones?", default=True
            ).ask()
            if use_terminal_plot is None:
                console.print("[red]Cancelled. Returning to main menu...[/red]")
                continue
            elif use_terminal_plot:
                try:
                    plot_last_loss_terminal()
                except FileNotFoundError:
                    console.print(
                        "[red]✘ No previous training run found. Please train a model first.[/red]"
                    )
            else:
                try:
                    plot_last_loss()
                except FileNotFoundError:
                    console.print(
                        "[red]✘ No previous training run found. Please train a model first.[/red]"
                    )
        elif choice == "Show predictions":
            use_terminal_plot = questionary.confirm(
                "Use terminal plots instead of graphical ones?", default=True
            ).ask()
            if use_terminal_plot is None:
                console.print("[red]Cancelled. Returning to main menu...[/red]")
                continue
            elif use_terminal_plot:
                try:
                    plot_last_predictions_terminal()
                except FileNotFoundError:
                    console.print("[red]✘ No predictions found. Please train a model first.[/red]")
            else:
                try:
                    plot_last_predictions()
                except FileNotFoundError:
                    console.print("[red]✘ No predictions found. Please train a model first.[/red]")


def main():
    clear_terminal()
    show_header()
    main_menu()


if __name__ == "__main__":
    main()
