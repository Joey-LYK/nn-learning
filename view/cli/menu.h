/**
 * menu.h — CLI 视图
 *
 * 只依赖 service/view_interface.h，与 service/capability/component 完全解耦。
 */

#ifndef VIEW_CLI_MENU_H
#define VIEW_CLI_MENU_H

#include "service/view_interface.h"

void cli_menu_run(ViewInterface *iface, void *ctx);

#endif /* VIEW_CLI_MENU_H */
