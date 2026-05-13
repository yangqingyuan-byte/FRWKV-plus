#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal vendored WeChat notifier for experiment completion messages.
"""
import requests


class WeChatNotifier:
    def __init__(self, method='serverchan', **kwargs):
        self.method = method
        if method == 'serverchan':
            self.sendkey = kwargs.get('sendkey')
            if not self.sendkey:
                raise ValueError("serverchan/xtuis requires sendkey")
        else:
            raise ValueError(f"Unsupported method: {method}")

    def send_serverchan(self, title, content):
        url = f"https://wx.xtuis.cn/{self.sendkey}.send"
        data = {"text": title, "desp": content}
        try:
            resp = requests.post(url, data=data, timeout=10)
            resp.raise_for_status()
            return True, "发送成功"
        except Exception as e:
            return False, f"请求异常: {e}"

    def send(self, title, content):
        return self.send_serverchan(title, content)
