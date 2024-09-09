# Copyright 2024 TBD Labs Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__version__ = "0.0.0"

__desc__ = """
Welcome to latched

Latched is an open source library for converting models to run on any device.
"""

# ANSI escape codes for colors
CYAN = "\033[96m"
WHITE = "\033[97m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"
BOLD = "\033[1m"

LATCHED_LOGO: str = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════╗
║ {WHITE}██╗      █████╗ ████████╗ ██████╗██╗  ██╗███████╗██████╗ {CYAN}║
║ {WHITE}██║     ██╔══██╗╚══██╔══╝██╔════╝██║  ██║██╔════╝██╔══██╗{CYAN}║
║ {WHITE}██║     ███████║   ██║   ██║     ███████║█████╗  ██║  ██║{CYAN}║
║ {WHITE}██║     ██╔══██║   ██║   ██║     ██╔══██║██╔══╝  ██║  ██║{CYAN}║
║ {WHITE}███████╗██║  ██║   ██║   ╚██████╗██║  ██║███████╗██████╔╝{CYAN}║
║ {WHITE}╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚══════╝╚═════╝ {CYAN}║
╚══════════════════════════════════════════════════════════╝
{RESET}"""

print(LATCHED_LOGO)
print(f"{__desc__}")
