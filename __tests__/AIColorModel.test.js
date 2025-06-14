const AIColorModel = require('../lib/AIColorModel');

describe('AIColorModel core methods', () => {
  let model;

  beforeEach(() => {
    model = new AIColorModel();
    // stub out persistence and heavy training
    model.saveToStorage = jest.fn();
    model.trainModel = jest.fn();
  });

  test('addCalibrationData validates input and stores data', () => {
    const good = model.addCalibrationData([0,0,0,0], [50,0,0]);
    expect(good).toBe(true);
    expect(model.calibrationData).toHaveLength(1);
    expect(model.trainModel).toHaveBeenCalled();
    const bad = model.addCalibrationData([0,0], [1,2,3]);
    expect(bad).toBe(false);
  });

  test('addLearningData stores learning entry and updates stats', async () => {
    model.trainModel = jest.fn();
    await model.addLearningData([0,0,0,0], [50,0,0], [40,0,0], true, 1.2);
    expect(model.learningData).toHaveLength(1);
    expect(model.realWorldResults).toHaveLength(1);
    expect(model.modelStats.totalAttempts).toBe(1);
    expect(model.modelStats.successfulMatches).toBe(1);
    expect(model.trainModel).toHaveBeenCalled();
  });

  test('trainModel aggregates samples and toggles training flag', async () => {
    model.trainNeuralNetwork = jest.fn();
    model.trainLinearModel = jest.fn();
    model.calibrationData = [{cmyk:[0,0,0,0], printedLab:[50,0,0]}];
    model.learningData = [{cmyk:[0,0,0,0], targetLab:[50,0,0], printedLab:[40,0,0], accurate:true}];
    await model.trainModel();
    expect(model.isTraining).toBe(false);
    expect(model.modelStats.totalSamples).toBe(2);
    expect(model.trainNeuralNetwork).toHaveBeenCalled();
    expect(model.trainLinearModel).toHaveBeenCalled();
  });

  test('predictCMYKAdjustment returns a suggestion object', async () => {
    const result = await model.predictCMYKAdjustment([50,0,0], [40,0,0], [0,0,0,0]);
    expect(result).toHaveProperty('suggested');
    expect(result.suggested).toHaveLength(4);
    expect(Array.isArray(result.suggested)).toBe(true);
  });
});
