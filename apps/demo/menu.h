/**
 * task_menu.h — 菜单渲染与用户交互
 *
 * 应用层的 UI 模块。负责渲染任务菜单、获取用户选择、分发执行。
 * 可以被替换为其他交互方式（如 CLI 参数、GUI）。
 */

#ifndef TASK_MENU_H
#define TASK_MENU_H

#include "apps/demo/tasks/task.h"
#include "trainer/trainer.h"

/**
 * 渲染菜单并等待用户选择。
 *
 * @param tasks     任务数组
 * @param count      任务数量
 * @param callback   进度回调（传递给 trainer）
 * @param user_data  回调用户数据
 * @return 用户选择（0=退出, 1~count=对应任务）, -1=无效输入
 */
int task_menu_run(Task **tasks, int count, TrainCallback callback, void *user_data);

#endif /* TASK_MENU_H */
