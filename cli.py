import os
from pyfiglet import figlet_format
from rich.console import Console
from rich.panel import Panel
import questionary
from trainer import train_model
from visualization import (
    plot_last_decision_boundary,
    plot_last_decision_boundary_terminal,
    plot_last_loss,
    plot_last_loss_terminal,
    plot_last_predictions,
    plot_last_predictions_terminal,
)

console = Console()


def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")


def show_header():
    ascii_banner = figlet_format("MLP Trainer", font="ansi_shadow", width=90)
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


def prompt_plotting_config(term_plot, gui_plot):
    use_terminal_plot = questionary.confirm(
        "Use terminal plots instead of graphical ones?", default=True
    ).ask()
    if use_terminal_plot is None:
        console.print("[red]Cancelled. Returning to main menu...[/red]")
        return None
    elif use_terminal_plot:
        try:
            return term_plot()
        except FileNotFoundError:
            console.print("[red]✘ No data found. Please train a model first.[/red]")
    else:
        try:
            return gui_plot()
        except FileNotFoundError:
            console.print("[red]✘ No data found. Please train a model first.[/red]")


def main_menu():
    while True:
        choice = questionary.select(
            "Choose an option:",
            choices=[
                "🧠 Train model",
                "📉 Show loss curves",
                "🔍 Show predictions",
                "🌈 Show decision boundary",
                "❌ Exit",
            ],
        ).ask()

        if choice == "❌ Exit":
            os.system("clear")
            console.print("[bold red]Goodbye![/bold red]")
            break
        elif choice == "🧠 Train model":
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
        elif choice == "📉 Show loss curves":
            prompt_plotting_config(plot_last_loss_terminal, plot_last_loss)
        elif choice == "🔍 Show predictions":
            prompt_plotting_config(plot_last_predictions_terminal, plot_last_predictions)
        elif choice == "🌈 Show decision boundary":
            prompt_plotting_config(
                plot_last_decision_boundary_terminal, plot_last_decision_boundary
            )


def main():
    clear_terminal()
    show_header()
    main_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]✘ Exiting due to keyboard interrupt.[/red]")
