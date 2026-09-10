#include "PainterEngine.h"
#include "stdio.h"


PX_ANN ann;
px_int epoch;
px_bool is_trained = PX_FALSE;  // 鏍囪鏄惁宸茬粡璁粌杩?

typedef struct
{
	px_point2D initv;
	px_float prediction;
	px_float truely;
}PX_Object_AnnBall;
PX_Object* PX_AnnBallCreate(px_float vx, px_float vy);

// 淇濆瓨绁炵粡缃戠粶鍒版枃浠?
px_bool PX_AnnSaveToFile(const px_char* filename)
{
	px_void* buffer = PX_NULL;
	px_int size = 0;
	
	// 鑾峰彇绁炵粡缃戠粶鏁版嵁澶у皬
	if (!PX_ANNExport(&ann, PX_NULL, &size))
	{
		printf("Failed to get ANN export size\n");
		return PX_FALSE;
	}
	
	// 鍒嗛厤鍐呭瓨
	buffer = MP_Malloc(mp, size);
	if (!buffer)
	{
		printf("Failed to allocate memory for ANN export\n");
		return PX_FALSE;
	}
	
	// 瀵煎嚭绁炵粡缃戠粶鏁版嵁
	if (!PX_ANNExport(&ann, buffer, &size))
	{
		printf("Failed to export ANN data\n");
		MP_Free(mp, buffer);
		return PX_FALSE;
	}
	
	// 淇濆瓨鍒版枃浠?
	if (PX_SaveDataToFile(buffer, size, filename) != 0)
	{
		printf("Failed to save ANN to file: %s\n", filename);
		MP_Free(mp, buffer);
		return PX_FALSE;
	}
	
	printf("ANN saved to file: %s (size: %d bytes)\n", filename, size);
	MP_Free(mp, buffer);
	return PX_TRUE;
}

// 浠庢枃浠跺姞杞界缁忕綉缁?
px_bool PX_AnnLoadFromFile(const px_char* filename)
{
	PX_IO_Data io = PX_LoadFileToIOData(filename);
	if (!io.size)
	{
		printf("Failed to load ANN file: %s\n", filename);
		return PX_FALSE;
	}
	
	// 鍏堥噴鏀剧幇鏈夌殑绁炵粡缃戠粶
	PX_ANNFree(&ann);
	
	// 浠庢枃浠跺鍏ョ缁忕綉缁?
	if (!PX_ANNImport(mp, &ann, io.buffer, io.size))
	{
		printf("Failed to import ANN from file: %s\n", filename);
		PX_FreeIOData(&io);
		return PX_FALSE;
	}
	
	printf("ANN loaded from file: %s (size: %d bytes)\n", filename, io.size);
	PX_FreeIOData(&io);
	is_trained = PX_TRUE;  // 鏍囪涓哄凡璁粌
	return PX_TRUE;
}

px_void PX_AnnTrainOnEpoch()
{
	px_double x=0, y=0;
	px_double vx = PX_randRange(50, 200);
	px_double vy = PX_randRange(50, 200);
	px_double initvx = vx;
	px_double initvy = vy;
	px_double input[2];
	px_double exp;
	px_double result;
	while (PX_TRUE)
	{
		vy -= 98 * 20.f / 1000.f;
		x += vx * 20.f / 1000.f;
		y += vy * 20.f / 1000.f;
		if (y<0)
		{
			break;
		}
	}
	exp = x / 1000.0;
	input[0] = initvx / 200.f;
	input[1] = initvy / 200.f;
	PX_ANNTrain(&ann, input, &exp);
	PX_ANNForward(&ann, input);
	PX_ANNGetOutput(&ann, &result);
	printf("diff:%lf\n",PX_ABS(result-exp));
}

px_int update_time=0;
px_float t;
PX_OBJECT_UPDATE_FUNCTION(AnnBallUpdate)
{
	PX_Object_AnnBall *pBall=PX_ObjectGetDescIndex(PX_Object_AnnBall,pObject,0);
	update_time+=elapsed*5;
	if (update_time>=20)
	{
		update_time -= 20;
		pObject->vy -= 98 * 20.f / 1000.f;
		pObject->x+=pObject->vx*20.f/1000.f;
		pObject->y+=pObject->vy*20.f/1000.f;
	}

	if (pObject->y < -100)
	{
		// 濡傛灉杩樻病鏈夎缁冭繃锛岃繘琛岃缁?
		if (!is_trained)
		{
			for (px_int i = 0; i < 1000; i++)
			{
				PX_AnnTrainOnEpoch();
			}
			epoch += 1;
			
			// 姣忚缁?0杞繚瀛樹竴娆?
			if (epoch % 10 == 0)
			{
				PX_AnnSaveToFile("ann_trained.px");
			}
		}
		PX_ObjectDelayDelete(pObject);
		PX_AnnBallCreate(PX_randRange(50, 200), PX_randRange(50, 200));
	}
}

PX_OBJECT_RENDER_FUNCTION(AnnBallRender)
{
	PX_Object_AnnBall* pBall = PX_ObjectGetDescIndex(PX_Object_AnnBall, pObject, 0);
	PX_GeoDrawBall(psurface, 32+pObject->x, 500-pObject->y-16, 16, PX_COLOR(255, 0, 0, 0));
	do
	{
		px_char content[64] = {0};
		if (is_trained)
		{
			PX_sprintf1(content, 64, "Trained ANN - epoch:%1k", PX_STRINGFORMAT_INT(epoch));
		}
		else
		{
			PX_sprintf1(content, 64, "Training... epoch:%1k", PX_STRINGFORMAT_INT(epoch));
		}
		PX_FontDrawText(psurface, 32 + pObject->x, 500 - pObject->y - 16-36,PX_ALIGN_MIDBOTTOM, content, PX_COLOR(255, 0, 0, 0));
	} while (0);

	PX_GeoDrawBall(psurface,32+pBall->prediction * 1000.0, 500-16, 16, PX_COLOR(255, 255, 0, 0));
	PX_FontDrawText(psurface, 32 + pBall->prediction * 1000.0, 500 - 16, PX_ALIGN_MIDBOTTOM, "prediction", PX_COLOR(255, 0, 0, 0));
	PX_GeoDrawBall(psurface, 32+pBall->truely * 1000.0, 500-16, 16, PX_COLOR(64, 0, 0, 255));
	PX_FontDrawText(psurface, 32 + pBall->truely * 1000.0, 500 - 16, PX_ALIGN_MIDBOTTOM, "actual", PX_COLOR(255, 0, 0, 0));
}

PX_Object* PX_AnnBallCreate(px_float vx, px_float vy)
{
	PX_Object* pObject = PX_ObjectCreateEx(mp, root, 0, 0, 0, 0, 0, 0, 0, AnnBallUpdate, AnnBallRender, 0,0,sizeof(PX_Object_AnnBall));
	PX_Object_AnnBall *pBall = PX_ObjectGetDescIndex(PX_Object_AnnBall, pObject, 0);
	px_double io[2];
	px_double result;
	px_float initvx = vx;
	px_float initvy = vy;


	pBall->initv.x = initvx;
	pBall->initv.y = initvy;
	pObject->vx= initvx;
	pObject->vy= initvy;

	io[0] = initvx / 200.f;
	io[1] = initvy / 200.f;
	PX_ANNForward(&ann, io);
	PX_ANNGetOutput(&ann, &result);
	pBall->prediction = result;

	do
	{
		px_double x = 0, y = 0;
		px_double vx = initvx;
		px_double vy = initvy;
		while (PX_TRUE)
		{
			vy -= 98 * 20.f / 1000.f;
			x += vx * 20.f / 1000.f;
			y += vy * 20.f / 1000.f;
			if (y < 0)
			{
				break;
			}
		}
		pBall->truely = x / 1000.0;
	} while (0);

	return pObject;
}

PX_OBJECT_RENDER_FUNCTION(grounddraw)
{
	PX_GeoDrawLine(psurface, 0, 500-16, 800, 500-16,3, PX_COLOR(128, 255, 0, 0));
}

px_int px_main()
{
	PainterEngine_Initialize(800, 600);
	PainterEngine_SetBackgroundColor(PX_COLOR_WHITE);
	
	// 灏濊瘯鍔犺浇宸茶缁冪殑绁炵粡缃戠粶
	if (PX_AnnLoadFromFile("ann_trained.px"))
	{
		printf("Successfully loaded trained ANN from file!\n");
		epoch = 100;  // 璁剧疆涓€涓悎鐞嗙殑epoch鍊?
	}
	else
	{
		printf("No trained ANN found, initializing new network...\n");
		// 鍒濆鍖栨柊鐨勭缁忕綉缁?
		PX_ANNInitialize(mp,&ann,0.005,PX_ANN_REGULARZATION_NONE,0);
		PX_ANNAddLayer(&ann,2,0,PX_ANN_ACTIVATION_FUNCTION_LINEAR,PX_ANN_LAYER_WEIGHT_INITMODE_GAUSSRAND,0.5);
		PX_ANNAddLayer(&ann,64,1,PX_ANN_ACTIVATION_FUNCTION_SIGMOID, PX_ANN_LAYER_WEIGHT_INITMODE_GAUSSRAND,0.5);
		PX_ANNAddLayer(&ann,128,1, PX_ANN_ACTIVATION_FUNCTION_SIGMOID, PX_ANN_LAYER_WEIGHT_INITMODE_GAUSSRAND,0.5);
		PX_ANNAddLayer(&ann,1,0, PX_ANN_ACTIVATION_FUNCTION_LINEAR, PX_ANN_LAYER_WEIGHT_INITMODE_GAUSSRAND,0.5);
		epoch = 0;
	}
	
	PX_AnnBallCreate(PX_randRange(50, 200), PX_randRange(50, 200));
	PX_ObjectCreateFunction(mp, root, 0, grounddraw, 0);
	
	return 0;
}
