// Extracted from ColorMatcherPro_CALIBRATION_FIXED_v2.1.8.html
const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

// Enhanced AI Color Model with proper data flow
class AIColorModel {
  constructor(options = {}) {
    const {
      deltaLThreshold = 0.5,
      deltaAThreshold = 0.5,
      deltaBThreshold = 0.5
    } = options;

    this.calibrationData = [];
    this.learningData = [];
    this.realWorldResults = [];
    this.neuralModel = null;
    this.linearModel = null;
    this.iccTransform = null;
    this.iccProfileName = '';
    this.iccProfileBase64 = '';
    this.isTraining = false;
    this.modelStats = {
      accuracy: 0,
      totalSamples: 0,
      neuralNetworkActive: false,
      confidence: 0,
      successfulMatches: 0,
      totalAttempts: 0
    };

    this.deltaLThreshold = deltaLThreshold;
    this.deltaAThreshold = deltaAThreshold;
    this.deltaBThreshold = deltaBThreshold;

    console.log('🤖 AIColorModel initialized');
  }

  // CRITICAL: Add calibration data with proper validation
  addCalibrationData(cmyk, printedLab) {
    console.log('🔍 Adding calibration data:', { cmyk, printedLab });
    
    if (!cmyk || !printedLab || cmyk.length !== 4 || printedLab.length !== 3) {
      console.error('❌ Invalid calibration data format');
      return false;
    }
    
    // Create a unique key for this CMYK combination
    const cmykKey = cmyk.join(',');
    
    // Find existing entry by CMYK values
    const existingIndex = this.calibrationData.findIndex(d => 
      d.cmyk.join(',') === cmykKey
    );
    
    const newEntry = { 
      cmyk: [...cmyk], 
      printedLab: [...printedLab],
      timestamp: Date.now(),
      cmykKey: cmykKey
    };
    
    if (existingIndex >= 0) {
      console.log('🔄 Updating existing calibration entry at index:', existingIndex);
      this.calibrationData[existingIndex] = newEntry;
    } else {
      console.log('✅ Adding new calibration entry');
      this.calibrationData.push(newEntry);
    }
    
    console.log('📊 Total calibration data entries:', this.calibrationData.length);
    
    // CRITICAL: Save immediately after each addition
    this.saveToStorage();
    
    // Retrain after every new calibration sample
    this.trainModel();
    
    return true;
  }

  // Add learning data from test results
  async addLearningData(cmyk, targetLab, printedLab, accurate, deltaE) {
    console.log('🧠 Adding learning data:', { cmyk, targetLab, printedLab, accurate, deltaE });
    
    const sample = {
      cmyk: [...cmyk],
      targetLab: [...targetLab],
      printedLab: [...printedLab],
      accurate,
      deltaE,
      timestamp: Date.now(),
      iterationId: Date.now() + Math.random()
    };
    
    this.learningData.push(sample);
    
    // Also store in real-world results for comprehensive tracking
    this.realWorldResults.push({
      ...sample,
      resultType: 'color_match',
      success: accurate,
      accuracy: deltaE <= 2.0 ? 'excellent' : deltaE <= 5.0 ? 'good' : 'needs_improvement'
    });

    // Update success statistics
    this.modelStats.totalAttempts++;
    if (accurate) {
      this.modelStats.successfulMatches++;
    }
    
    console.log('📈 Learning data added. Total learning samples:', this.learningData.length);
    
    // Auto-save after each learning iteration
    this.saveToStorage();

    // Retrain with combined data
    await this.trainModel();
  }

  // CRITICAL: Train model on BOTH calibration + learning data
  async trainModel() {
    if (this.isTraining) {
      console.log('⏳ Training already in progress, skipping...');
      return;
    }
    
    this.isTraining = true;
    console.log('🧠 Starting AI training...');

    try {
      // CRITICAL: Combine calibration and learning data properly
      const allSamples = [
        // Calibration data: CMYK -> measured LAB (foundation)
        ...this.calibrationData.map(c => ({
          cmyk: c.cmyk,
          targetLab: c.printedLab,
          printedLab: c.printedLab,
          weight: 1.0 // Full weight for calibration data
        })),
        // Learning data: CMYK -> printed LAB vs target LAB (refinement)
        ...this.learningData.map(l => ({
          cmyk: l.cmyk,
          targetLab: l.targetLab,
          printedLab: l.printedLab,
          weight: l.accurate ? 1.2 : 0.8 // Higher weight for accurate results
        }))
      ];

      this.modelStats.totalSamples = allSamples.length;
      console.log(`🧠 Training on ${allSamples.length} samples (${this.calibrationData.length} calibration + ${this.learningData.length} learning)`);

      if (allSamples.length > 0) {
        await this.trainNeuralNetwork(allSamples);
        this.modelStats.neuralNetworkActive = true;
      }

      // Always maintain linear fallback
      this.trainLinearModel(allSamples);

      // Calculate overall accuracy
      if (this.modelStats.totalAttempts > 0) {
        this.modelStats.accuracy = (this.modelStats.successfulMatches / this.modelStats.totalAttempts) * 100;
      }

      console.log('✅ Training completed successfully');

    } catch (error) {
      console.error('❌ Training error:', error);
    } finally {
      this.isTraining = false;
    }
  }

  // Train neural network with TensorFlow.js
  async trainNeuralNetwork(samples) {
    try {
      console.log('🧠 Training neural network with', samples.length, 'samples');
      
      // Prepare training data with proper feature engineering
      const inputs = samples.map(s => [
        ...s.cmyk,           // 4 features: CMYK values
        ...s.targetLab,      // 3 features: target LAB
        // Add derived features for better learning
        s.cmyk[0] + s.cmyk[1] + s.cmyk[2] + s.cmyk[3], // Total ink coverage
        Math.abs(s.targetLab[1]) + Math.abs(s.targetLab[2]), // Color saturation
      ]); // 9 total input features
      
      const outputs = samples.map(s => s.printedLab); // 3 outputs: printed LAB
      const weights = samples.map(s => s.weight || 1.0); // Sample weights

      // Create or update neural network
      if (!this.neuralModel) {
        this.neuralModel = tf.sequential({
          layers: [
            tf.layers.dense({ inputShape: [9], units: 32, activation: 'relu' }),
            tf.layers.dropout({ rate: 0.2 }),
            tf.layers.dense({ units: 24, activation: 'relu' }),
            tf.layers.dropout({ rate: 0.1 }),
            tf.layers.dense({ units: 16, activation: 'relu' }),
            tf.layers.dense({ units: 3 }) // L, a, b output
          ]
        });

        this.neuralModel.compile({
          optimizer: tf.train.adam(0.001),
          loss: 'meanSquaredError',
          metrics: ['mae']
        });
      }

      // Train the model with sample weights
      const xs = tf.tensor2d(inputs);
      const ys = tf.tensor2d(outputs);
      const sampleWeights = tf.tensor1d(weights);

      const history = await this.neuralModel.fit(xs, ys, {
        epochs: Math.min(100, Math.max(20, samples.length * 2)),
        batchSize: Math.min(32, Math.max(4, Math.floor(samples.length / 3))),
        validationSplit: samples.length > 20 ? 0.2 : 0,
        sampleWeight: sampleWeights,
        verbose: 0
      });

      // Calculate accuracy based on final loss
      const finalLoss = history.history.loss[history.history.loss.length - 1];
      const networkAccuracy = Math.max(0, Math.min(100, (1 - Math.sqrt(finalLoss) / 50) * 100));
      this.modelStats.confidence = Math.min(100, networkAccuracy + (samples.length * 0.5));

      console.log(`🧠 Neural network trained: Loss=${finalLoss.toFixed(4)}, Confidence=${this.modelStats.confidence.toFixed(1)}%`);

      xs.dispose();
      ys.dispose();
      sampleWeights.dispose();

    } catch (error) {
      console.error('❌ Neural network training error:', error);
      this.modelStats.neuralNetworkActive = false;
    }
  }

  // Train linear model fallback
  trainLinearModel(samples) {
    try {

      console.log('📊 Training linear model with', samples.length, 'samples');

      // Prepare matrices for weighted linear regression
      const X = samples.map(s => [
        1, // bias term
        ...s.cmyk,
        ...s.targetLab,
        // Add interaction terms for better modeling
        s.cmyk[0] * s.targetLab[0], // C * L
        s.cmyk[1] * s.targetLab[1], // M * a
        s.cmyk[2] * s.targetLab[2], // Y * b
        s.cmyk[3] * s.targetLab[0], // K * L
        // Add quadratic terms
        s.cmyk[0] * s.cmyk[0], // C²
        s.cmyk[1] * s.cmyk[1], // M²
        s.cmyk[2] * s.cmyk[2], // Y²
        s.cmyk[3] * s.cmyk[3], // K²
      ]);

      const Y = samples.map(s => s.printedLab);
      const W = samples.map(s => s.weight || 1.0);

      // Apply sample weights to X and Y
      const weightedX = X.map((row, i) => row.map(val => val * Math.sqrt(W[i])));
      const weightedY = Y.map((row, i) => row.map(val => val * Math.sqrt(W[i])));

      // Solve for each LAB component
      const coefficients = [];
      for (let i = 0; i < 3; i++) {
        const y = weightedY.map(row => row[i]);
        try {
          const XtX = numeric.dot(numeric.transpose(weightedX), weightedX);
          const Xty = numeric.dot(numeric.transpose(weightedX), y);
          
          // Add regularization to prevent overfitting
          const lambda = 0.01;
          for (let j = 0; j < XtX.length; j++) {
            XtX[j][j] += lambda;
          }
          
          const coeff = numeric.solve(XtX, Xty);
          coefficients.push(coeff);
        } catch (e) {
          console.warn(`Linear regression failed for component ${i}, using fallback`);
          coefficients.push(new Array(weightedX[0].length).fill(0));
        }
      }

      this.linearModel = coefficients;
      console.log(`📊 Linear model trained with ${samples.length} samples`);

    } catch (error) {
      console.error('❌ Linear model training error:', error);
    }
  }

  // CRITICAL: Predict CMYK adjustments using trained models
async predictCMYKAdjustment(targetLab, printedLab, currentCmyk) {
    console.log('🔍 Predicting CMYK adjustment:', { targetLab, printedLab, currentCmyk });

    try {
      let baseCmyk = null;
      if (this.iccTransform) {
        try {
          baseCmyk = this.iccTransform.apply(targetLab);
          currentCmyk = baseCmyk;
        } catch (e) {
          console.warn('⚠️ ICC transform failed:', e);
        }
      }
      // Always attempt to predict the printed LAB values
      const hasMeasurement = printedLab && printedLab.some(v => v !== 0);
      let predictedLab;

      if (this.neuralModel && this.modelStats.neuralNetworkActive) {
          console.log('🧠 Using neural network for prediction');
          const features = [
            ...currentCmyk,
            ...targetLab,
            currentCmyk[0] + currentCmyk[1] + currentCmyk[2] + currentCmyk[3],
            Math.abs(targetLab[1]) + Math.abs(targetLab[2])
          ];

          const input = tf.tensor2d([features]);
          const prediction = this.neuralModel.predict(input);
          const predictionData = await prediction.data();
          predictedLab = Array.from(predictionData);

          input.dispose();
          prediction.dispose();
        } else if (this.linearModel && this.linearModel.length === 3) {
          console.log('📊 Using linear model for prediction');
          // Use linear model fallback
          const features = [
            1, // bias
            ...currentCmyk,
            ...targetLab,
            currentCmyk[0] * targetLab[0],
            currentCmyk[1] * targetLab[1],
            currentCmyk[2] * targetLab[2],
            currentCmyk[3] * targetLab[0],
            currentCmyk[0] * currentCmyk[0],
            currentCmyk[1] * currentCmyk[1],
            currentCmyk[2] * currentCmyk[2],
            currentCmyk[3] * currentCmyk[3]
          ];

          predictedLab = this.linearModel.map(coeff =>
            coeff.reduce((sum, c, i) => sum + c * (features[i] || 0), 0)
          );
        } else if (hasMeasurement) {
          console.log('📏 Using measured LAB values for error calculation');
          predictedLab = [...printedLab];
        } else {
          console.log('🎨 Using color theory fallback');
          predictedLab = [...targetLab];
        }
      }

      // Calculate LAB error using either prediction or measurement
      const labError = [
        targetLab[0] - predictedLab[0], // ΔL
        targetLab[1] - predictedLab[1], // Δa
        targetLab[2] - predictedLab[2]  // Δb
      ];

      // Convert LAB error to CMYK adjustments using enhanced color theory
      const cmykAdjustment = this.labErrorToCMYKAdjustment(labError, currentCmyk, targetLab);

      // Calculate raw suggestion and find a single scale factor to keep all channels within bounds
      const rawSug = currentCmyk.map((v, i) => v + cmykAdjustment[i]);
      let s = 1;
      for (let i = 0; i < 4; i++) {
        const diff = rawSug[i] - currentCmyk[i];
        if (diff > 0) {
          s = Math.min(s, (100 - currentCmyk[i]) / diff);
        } else if (diff < 0) {
          s = Math.min(s, (0 - currentCmyk[i]) / diff);
        }
      }

      // Apply scaled adjustments and clamp final values
      const suggestedCmyk = currentCmyk.map((v, i) => {
        const rawDelta = cmykAdjustment[i] * s;
        const unclamped = v + rawDelta;
        const final = clamp(unclamped, 0, 100);
        console.log({ index: i, rawDelta, unclamped, final });
        return Math.round(final);
      });

      const result = {
        suggested: suggestedCmyk,
        confidence: this.modelStats.confidence,
        method: this.modelStats.neuralNetworkActive ? 'Neural Network' : 
               this.linearModel ? 'Linear Model' : 'Color Theory',
        labError: labError,
        adjustment: cmykAdjustment
      };

      console.log('✅ Prediction completed:', result);
      return result;

    } catch (error) {
      console.error('❌ Prediction error:', error);
      return {
        suggested: [...currentCmyk],
        confidence: 0,
        method: 'Fallback',
        labError: [0, 0, 0],
        adjustment: [0, 0, 0, 0]
      };
    }
  }

  // Convert LAB error to CMYK adjustments
  labErrorToCMYKAdjustment(labError, currentCmyk, targetLab) {
    const [deltaL, deltaA, deltaB] = labError;
    
    // Enhanced color theory adjustments with adaptive scaling
    const adjustments = [0, 0, 0, 0]; // C, M, Y, K
    
    // Calculate error magnitude for adaptive scaling
    const errorMagnitude = Math.sqrt(deltaL*deltaL + deltaA*deltaA + deltaB*deltaB);
    const scaleFactor = Math.min(2.0, Math.max(0.1, errorMagnitude / 10));

    // Lightness adjustments (L* axis)
    if (Math.abs(deltaL) > this.deltaLThreshold) {
      const lightnessFactor = deltaL * 0.4 * scaleFactor;
      
      // Primary adjustment through K (black)
      adjustments[3] -= lightnessFactor;
      
      // Secondary adjustments through CMY if needed
      if (Math.abs(lightnessFactor) > 5) {
        const overflow = (Math.abs(lightnessFactor) - 5) * 0.3;
        adjustments[0] -= overflow * Math.sign(lightnessFactor);
        adjustments[1] -= overflow * Math.sign(lightnessFactor);
        adjustments[2] -= overflow * Math.sign(lightnessFactor);
      }
    }

    // Red/Green adjustments (a* axis)
    if (Math.abs(deltaA) > this.deltaAThreshold) {
      const aFactor = deltaA * 0.5 * scaleFactor;
      adjustments[1] += aFactor; // Magenta primarily affects a*
      adjustments[0] -= aFactor * 0.3; // Cyan secondary effect
    }

    // Yellow/Blue adjustments (b* axis)
    if (Math.abs(deltaB) > this.deltaBThreshold) {
      const bFactor = deltaB * 0.5 * scaleFactor;
      adjustments[2] += bFactor; // Yellow primarily affects b*
      adjustments[0] -= bFactor * 0.2; // Cyan secondary effect
    }

    // Apply learning from historical data if available
    if (this.learningData.length > 0) {
      const recentLearning = this.learningData.slice(-10); // Last 10 iterations
      const avgAdjustment = this.calculateHistoricalAdjustment(recentLearning, labError, targetLab);
      
      // Blend with color theory (70% theory, 30% learned)
      for (let i = 0; i < 4; i++) {
        adjustments[i] = adjustments[i] * 0.7 + avgAdjustment[i] * 0.3;
      }
    }

    // Limit adjustment magnitude to prevent extreme changes
    const maxAdjustment = 15; // Maximum 15% change per iteration
    return adjustments.map(adj => Math.max(-maxAdjustment, Math.min(maxAdjustment, adj)));
  }

  // Calculate historical adjustment patterns
  calculateHistoricalAdjustment(recentData, currentError, targetLab) {
    if (recentData.length === 0) return [0, 0, 0, 0];

    const adjustments = [0, 0, 0, 0];
    let totalWeight = 0;

    recentData.forEach(data => {
      // Find similar color targets (within LAB tolerance)
      const colorSimilarity = 1 / (1 + Math.sqrt(
        Math.pow(data.targetLab[0] - targetLab[0], 2) +
        Math.pow(data.targetLab[1] - targetLab[1], 2) +
        Math.pow(data.targetLab[2] - targetLab[2], 2)
      ));

      if (colorSimilarity > 0.1) { // Only use reasonably similar colors
        const weight = colorSimilarity * (data.accurate ? 2.0 : 0.5);
        
        // Calculate implicit adjustment (simplified)
        const errorDirection = [
          data.targetLab[0] - data.printedLab[0],
          data.targetLab[1] - data.printedLab[1],
          data.targetLab[2] - data.printedLab[2]
        ];

          // Convert error direction to CMYK adjustment direction
          adjustments[0] -= errorDirection[1] * 0.2 * weight; // a* -> Cyan
          adjustments[1] += errorDirection[1] * 0.4 * weight; // a* -> Magenta
          adjustments[2] += errorDirection[2] * 0.4 * weight; // b* -> Yellow
          adjustments[3] += -errorDirection[0] * 0.3 * weight; // L* -> Black

        totalWeight += weight;
      }
    });

    return totalWeight > 0 ? adjustments.map(adj => adj / totalWeight) : [0, 0, 0, 0];
  }

  // Get model statistics
  getModelStats() {
    return {
      ...this.modelStats,
      calibrationPatches: this.calibrationData.length,
      learningIterations: this.learningData.length,
      realWorldResults: this.realWorldResults.length,
      isTraining: this.isTraining,
      iccLoaded: !!this.iccTransform,
      successRate: this.modelStats.totalAttempts > 0 ?
        (this.modelStats.successfulMatches / this.modelStats.totalAttempts * 100) : 0
    };
  }

  // CRITICAL: Enhanced Save/Load functionality with comprehensive data
  saveToStorage() {
    const data = {
      calibrationData: this.calibrationData,
      learningData: this.learningData,
      realWorldResults: this.realWorldResults,
      modelStats: this.modelStats,
      iccProfileName: this.iccProfileName,
      iccProfileBase64: this.iccProfileBase64,
      version: '2.1.8',
      timestamp: Date.now(),
      metadata: {
        totalSessions: (JSON.parse(localStorage.getItem('colorMatcherAI') || '{}').metadata?.totalSessions || 0) + 1,
        lastSaved: new Date().toISOString()
      }
    };
    
    try {
      localStorage.setItem('colorMatcherAI', JSON.stringify(data));
      console.log(`✅ Data saved successfully: ${this.calibrationData.length} calibration + ${this.learningData.length} learning + ${this.realWorldResults.length} results`);
    } catch (error) {
      console.error('❌ Error saving data:', error);
    }
  }

  loadFromStorage() {
    try {
      const saved = localStorage.getItem('colorMatcherAI');
      if (saved) {
        const data = JSON.parse(saved);
        
        this.calibrationData = data.calibrationData || [];
        this.learningData = data.learningData || [];
        this.realWorldResults = data.realWorldResults || [];
        this.modelStats = { ...this.modelStats, ...data.modelStats };
        this.iccProfileName = data.iccProfileName || '';
        this.iccProfileBase64 = data.iccProfileBase64 || '';
        if (this.iccProfileBase64) {
          try {
            const binary = atob(this.iccProfileBase64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            this.loadICCProfile(bytes.buffer, this.iccProfileName);
          } catch (err) {
            console.error('❌ Failed to restore ICC profile:', err);
          }
        }
        
        console.log(`✅ Data loaded successfully: ${this.calibrationData.length} calibration + ${this.learningData.length} learning + ${this.realWorldResults.length} results`);
        
        // Retrain if we have data
        if (this.calibrationData.length > 0 || this.learningData.length > 0) {
          this.trainModel();
        }
        
        return true;
      }
    } catch (error) {
      console.error('❌ Error loading saved data:', error);
    }
    return false;
  }

  // Load ICC profile and create Lab->CMYK transform
  loadICCProfile(arrayBuffer, name = '') {
    try {
      const profile = lcmsjs.parse(arrayBuffer);
      this.iccTransform = lcmsjs.buildTransform(profile, 'lab', 'cmyk');
      this.iccProfileName = name;
      this.iccProfileBase64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
      console.log('✅ ICC profile loaded');
      return true;
    } catch (err) {
      console.error('❌ Failed to load ICC profile:', err);
      this.iccTransform = null;
      this.iccProfileName = '';
      this.iccProfileBase64 = '';
      return false;
    }
  }
}
module.exports = AIColorModel;
