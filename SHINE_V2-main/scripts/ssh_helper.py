#!/usr/bin/env python3
"""
SSH helper script using paramiko for password-based and key-based SSH.
Fully replaces sshpass for cluster management.

Usage:
    ssh_helper.py <user> <host> <password> <command>
    ssh_helper.py --check <user> <host> <password>
    ssh_helper.py --key <keyfile> <user> <host> <command>
    ssh_helper.py --check --key <keyfile> <user> <host>
"""
import sys
import argparse
import paramiko
import socket


def ssh_exec(user, host, password=None, key_filename=None, command=None, timeout=30, port=22):
    """Execute a command on a remote host via SSH."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        connect_kwargs = dict(
            hostname=host,
            port=port,
            username=user,
            timeout=timeout,
        )
        if key_filename:
            connect_kwargs["key_filename"] = key_filename
        else:
            connect_kwargs["password"] = password
            connect_kwargs["look_for_keys"] = False
            connect_kwargs["allow_agent"] = False

        client.connect(**connect_kwargs)

        if command == "__CHECK__":
            print("ok")
            return 0

        # Use get_transport().open_session() for nohup support
        transport = client.get_transport()
        channel = transport.open_session()
        channel.exec_command(command)

        # Read output
        stdout = channel.makefile("r")
        stderr = channel.makefile_stderr("r")

        for line in stdout:
            print(line, end="")
        for line in stderr:
            print(line, end="", file=sys.stderr)

        exit_code = channel.recv_exit_status()
        return exit_code
    except paramiko.AuthenticationException:
        print(f"Error: Authentication failed for {user}@{host}", file=sys.stderr)
        return 1
    except (paramiko.SSHException, socket.error) as e:
        print(f"Error: Cannot connect to {user}@{host}: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="SSH helper using paramiko")
    parser.add_argument("--check", action="store_true", help="Connectivity check mode")
    parser.add_argument("--key", type=str, default=None, help="Path to SSH private key file")
    parser.add_argument("--port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("args", nargs="+", help="user host [password] [command]")

    parsed = parser.parse_args()

    port = parsed.port

    if parsed.key:
        # Key-based auth: args = [user, host, command?]
        if len(parsed.args) < 2:
            parser.error("Key mode requires: user host [command]")
        user, host = parsed.args[0], parsed.args[1]
        if parsed.check:
            rc = ssh_exec(user, host, key_filename=parsed.key, command="__CHECK__", timeout=5, port=port)
        else:
            command = parsed.args[2] if len(parsed.args) > 2 else "echo ok"
            rc = ssh_exec(user, host, key_filename=parsed.key, command=command, port=port)
    else:
        # Password-based auth: args = [user, host, password, command?]
        if len(parsed.args) < 3:
            parser.error("Password mode requires: user host password [command]")
        user, host, password = parsed.args[0], parsed.args[1], parsed.args[2]
        if parsed.check:
            rc = ssh_exec(user, host, password=password, command="__CHECK__", timeout=5, port=port)
        else:
            command = parsed.args[3] if len(parsed.args) > 3 else "echo ok"
            rc = ssh_exec(user, host, password=password, command=command, port=port)

    sys.exit(rc)


if __name__ == "__main__":
    main()
