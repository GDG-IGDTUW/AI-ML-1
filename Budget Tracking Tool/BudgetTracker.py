import csv
import os

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import shutil
from datetime import datetime

DATA_FILE = 'budget_data.csv'

BACKUP_FILE = 'budget_data_backup.csv'

def initialize_data_file():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Type', 'Category', 'Amount', 'Description'])  # TODO: Add a timestamp field to the CSV header.
import os
import platform

def clear_screen():
    system = platform.system()
    if system == "Windows":
        os.system("cls")
    else:
        os.system("clear")

from datetime import datetime

def add_transaction(transaction_type, category, amount, description="No description"):
    if not isinstance(amount, (int, float)) or amount < 0:
        print("Invalid amount")
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(DATA_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([transaction_type, category, amount, description, timestamp])
    
    print("Transaction added successfully!")

def update_transaction(index, transaction_type, category, amount, description):
    # TODO: Validate that the provided index is valid.
    total=read_transactions()
    if len(total)==0:
        print("no transaction exists to be deleted")
        return
    if index<0 or index>=len(total):
        print("invalid index....please enter a valid index")
        return
    temp_file = DATA_FILE + '.tmp'
    with open(DATA_FILE, mode='r') as file, open(temp_file, mode='w', newline='') as temp:
        reader = csv.reader(file)
        writer = csv.writer(temp)
        for i, row in enumerate(reader):
            if i == index + 1:  # To skip header
                writer.writerow([transaction_type, category, amount, description])
            else:
                writer.writerow(row)
    os.replace(temp_file, DATA_FILE)

def delete_transaction(index):
    # TODO: Validate that the provided index is valid.
    total=read_transactions()
    if len(total)==0:
        print("no transaction exists to be deleted")
        return
    if index<0 or index>=len(total):
        print("invalid index....please enter a valid index")
        return
    temp_file = DATA_FILE + '.tmp'
    with open(DATA_FILE, mode='r') as file, open(temp_file, mode='w', newline='') as temp:
        reader = csv.reader(file)
        writer = csv.writer(temp)
        for i, row in enumerate(reader):
            if i != index + 1:  
                writer.writerow(row)
    os.replace(temp_file, DATA_FILE)

def read_transactions():
    with open(DATA_FILE, mode='r') as file:
        reader = csv.reader(file)
        return list(reader)[1:]  

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
def generate_report():
    # TODO: Extend the report to show income and expense totals by category.
    # TODO: Add functionality to generate a summary of transactions within a specific date range.
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    income_by_category = {}
    expenses_by_category = {}
    total_income = 0.0
    total_expenses = 0.0

    with open(DATA_FILE, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            transaction_date = datetime.strptime(row[4], '%Y-%m-%d %H:%M:%S')  # Assuming the date is in this format
            if start_date <= transaction_date <= end_date:
                category = row[1]
                amount = float(row[2])
                if row[0] == 'income':
                    total_income += amount
                    income_by_category[category] = income_by_category.get(category, 0) + amount
                elif row[0] == 'expense':
                    total_expenses += amount
                    expenses_by_category[category] = expenses_by_category.get(category, 0) + amount

    savings = total_income - total_expenses

    print("\nIncome by Category:")
    for category, amount in income_by_category.items():
        print(f"{category}: ${amount:.2f}")

    print("\nExpenses by Category:")
    for category, amount in expenses_by_category.items():
        print(f"{category}: ${amount:.2f}")

    print("\nOverall Summary:")
    print(f"Total Income: ${total_income:.2f}")
    print(f"Total Expenses: ${total_expenses:.2f}")
    print(f"Savings: ${savings:.2f}")
def export_to_excel_or_pdf():
    df = pd.read_csv(DATA_FILE)
    print("1. Export to Excel")
    print("2. Export to PDF")
    choice = input("Choose an option: ")
    if choice == '1':
        df.to_excel('budget_data.xlsx', index=False)
        print("Data exported to budget_data.xlsx")
    elif choice == '2':
        pdf_file = 'budget_data.pdf'
        c = canvas.Canvas(pdf_file, pagesize=letter)
        width, height = letter
        c.drawString(100, height - 40, "Budget Transactions Report")
        y_position = height - 80
        
        for i, row in df.iterrows():
            line = f"{row['Type']}, {row['Category']}, {row['Amount']}, {row['Description']}"
            c.drawString(100, y_position, line)
            y_position -= 20
            if y_position < 50:
                c.showPage()
                y_position = height - 80
        
        c.save()
        print("Data exported to budget_data.pdf")
    else:
        print("Invalid choice.")

def search_transactions():
    print("\nSearch by:")
    print("1. Category")
    print("2. Type (Income/Expense)")
    print("3. Description")
    
    search_choice = input("Choose a search criterion (1, 2, or 3): ")

    search_term = input("Enter search term: ").lower()

    found_transactions = []

    with open(DATA_FILE, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if search_choice == '1' and search_term in row[1].lower():  
                found_transactions.append(row)
            elif search_choice == '2' and search_term in row[0].lower(): 
                found_transactions.append(row)
            elif search_choice == '3' and search_term in row[3].lower(): 
                found_transactions.append(row)

    if found_transactions:
        print("\nSearch Results:")
        for transaction in found_transactions:
            print(f"Type: {transaction[0]}, Category: {transaction[1]}, Amount: ${transaction[2]}, Description: {transaction[3]}")
    else:
        print("No transactions found matching your criteria.")
pass

    # TODO: Implement an "Undo" feature to revert the most recent addition, update, or deletion of a transaction.
def backup_data_file():
    """Creates a backup of the data file before modification."""
    if os.path.exists(DATA_FILE):
        shutil.copy(DATA_FILE, BACKUP_FILE)
    
def undo_last_action():
    """Restores the last saved state from the backup file."""
    if os.path.exists(BACKUP_FILE):
        shutil.copy(BACKUP_FILE, DATA_FILE)
        print("Last action undone successfully.")
    else:
        print("No previous state available to undo.")
    pass

def clear_screen():
    # TODO: Add functionality to clear the terminal screen at the start of each menu display.
    pass

def main():
    initialize_data_file()
    while True:
        # TODO: Call clear_screen() here for better readability.
        clear_screen()
        print("\nBudget Tracker")
        print("1. Add Transaction")
        print("2. Update Transaction")
        print("3. Delete Transaction")
        print("4. Generate Report")
        print("5. Search Transactions")  # TODO: Add this menu option for searching transactions.
        print("6. Export Data")  # TODO: Add this menu option for exporting data.
        print("7. Undo Last Action")  # TODO: Add this menu option for the undo feature.
        print("8. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            transaction_type = input("Enter transaction type (income/expense): ")
            category = input("Enter category: ")
            amount = input("Enter amount: ")
            description = input("Enter description (optional): ").strip() or "No description"  # Handle optional descriptions
            add_transaction(transaction_type, category, amount, description)
        elif choice == '2':
            total = read_transactions()
            if len(total) == 0:
                print("No transactions made yet. Please add transactions first.")
                continue
            else:
                while True:
                    index = int(input("Enter transaction index to update: "))  # TODO: Use a validated index.
                    if index >= 0 and index < len(total):
                        break
                    else:
                        print("Invalid index. Please enter a valid index.")

            transaction_type = input("Enter transaction type (income/expense): ")
            category = input("Enter category: ")
            amount = input("Enter amount: ")
            description = input("Enter description (optional): ").strip() or "No description"  # Handle optional descriptions
            update_transaction(index, transaction_type, category, amount, description)
        elif choice == '3':
            total = read_transactions()
            if len(total) == 0:
                print("No transactions to delete. Please first add transactions.")
                continue
            else:
                while True:
                    index = int(input("Enter transaction index to delete: "))  # TODO: Use a validated index.
                    if index >= 0 and index < len(total):
                        break
                    else:
                        print("Invalid index. Please enter again.")
                delete_transaction(index)
        elif choice == '4':
            generate_report()
        elif choice == '5':
            search_transactions()  # TODO: Implement this function.
        elif choice == '6':
            export_to_excel_or_pdf()  # TODO: Implement this function.
        elif choice == '7':
            undo_last_action()  # TODO: Implement this function.
        elif choice == '8':
            break
        else:
            print("Invalid choice. Please try again.")