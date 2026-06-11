/**
 * menu.h — CLI 视图：菜单渲染与用户交互
 */

#ifndef VIEW_CLI_MENU_H
#define VIEW_CLI_MENU_H

#include "service/training_service.h"
#include "capability/trainer.h"

int cli_menu_run(TrainingService *svc, TrainCallback callback, void *user_data);

#endif /* VIEW_CLI_MENU_H */
