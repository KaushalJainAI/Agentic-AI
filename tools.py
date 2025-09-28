import os
from dotenv import load_dotenv
from datetime import datetime
from langchain.tools import tool
import sympy as sp
import numpy as np
from langchain.tools import tavily_search


load_dotenv()


@tool
def get_system_time():
    """This tool provide correct system time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool 
def calculator():
    """This tool provides logic for doing basic calculations"""
    x = sp.symbols('x')
    y = sp.symbols('y')
    
        

 