# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import os
import time
from typing import Dict, Optional, Tuple

try:
    import websockets.sync.client
except ModuleNotFoundError:
    websockets = None

try:
    import websocket
except ModuleNotFoundError:
    websocket = None
from typing_extensions import override

from . import msgpack_numpy


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = 10093, api_key: Optional[str] = None) -> None:
        # 0.0.0.0 cannot be used as a connection target, here default 127.0.0.1
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _connect_websockets_sync(self, headers: Optional[Dict[str, str]]):
        if websockets is None:
            raise ModuleNotFoundError("websockets")
        connect_kwargs = {
            "compression": None,
            "max_size": None,
            "open_timeout": 150,
            "ping_interval": None,
            "ping_timeout": 60,
        }
        if headers:
            connect_kwargs["additional_headers"] = headers
        try:
            return websockets.sync.client.connect(self._uri, **connect_kwargs)
        except TypeError:
            # Older websockets.sync.client.connect versions do not accept
            # ping_* kwargs and use extra_headers instead of additional_headers.
            connect_kwargs.pop("ping_interval", None)
            connect_kwargs.pop("ping_timeout", None)
            if "additional_headers" in connect_kwargs:
                connect_kwargs["extra_headers"] = connect_kwargs.pop("additional_headers")
            return websockets.sync.client.connect(self._uri, **connect_kwargs)

    def _connect_websocket_client(self, headers: Optional[Dict[str, str]]):
        if websocket is None:
            raise ModuleNotFoundError("websocket-client")
        header_list = [f"{key}: {value}" for key, value in headers.items()] if headers else None
        return websocket.create_connection(
            self._uri,
            timeout=150,
            enable_multithread=False,
            header=header_list,
            suppress_origin=True,
        )

    def _wait_for_server(self, timeout: float = 300) -> Tuple[object, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        start_time = time.time()

        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to server within {timeout} seconds")

            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                try:
                    conn = self._connect_websockets_sync(headers)
                except ModuleNotFoundError:
                    conn = self._connect_websocket_client(headers)
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, OSError):
                logging.info(f"Still waiting for server {self._uri} ...")
                time.sleep(2)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass

    @override
    def predict_action(self, query_info: Dict) -> Dict:
        data = self._packer.pack(query_info)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)
