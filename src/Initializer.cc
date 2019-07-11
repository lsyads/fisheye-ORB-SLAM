/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>
#include<cmath>

namespace ORB_SLAM2
{

Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    mK = ReferenceFrame.mK.clone();
    mDistCoef = ReferenceFrame.mDistCoef.clone();
    mfAlpha = mDistCoef.at<float>(0);
    mfBeta = mDistCoef.at<float>(1);
    
    mR2range = 1.0f/(mfBeta*(2*mfAlpha-1));
    
    //cout<<"mR2range: "<<mR2range<<endl;
    
    mvKeys1 = ReferenceFrame.mvKeysUn;
    
    mvP3M1 = ReferenceFrame.mvP3M;

    mSigma = sigma;
    mSigma2 = 1.0f/(sigma*sigma);
    mMaxIterations = iterations;

    //max of FoV is 150 degrees, so we can find the solution of the equation
    const float coefA = 13.928*mfAlpha*mfAlpha*mfBeta+2*mfAlpha-1;  //150 degrees coefiicient of equation
    const float coefB = 2-2*mfAlpha;
    const float coefC = -1;

    mzMin = (-coefB + sqrt(coefB*coefB-4*coefA*coefC)) / (2*coefA)-0.1;
}

bool Initializer::Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12, cv::Mat &R21, cv::Mat &t21,
                             vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    mvKeys2 = CurrentFrame.mvKeysUn;
    
    mvP3M2 = CurrentFrame.mvP3M;
    
    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());
    mvbMatched1.resize(mvKeys1.size());
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        if(vMatches12[i]>=0)
        {
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
            mvbMatched1[i]=true;
        }
        else
            mvbMatched1[i]=false;
    }

    const int N = mvMatches12.size();

    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

    DUtils::Random::SeedRandOnce(0);

    for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            int idx = vAvailableIndices[randi];

            mvSets[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    float SH, SF;
    cv::Mat H, F;

    thread threadH(&Initializer::FindHomography,this,ref(vbMatchesInliersH), ref(SH), ref(H));
    thread threadF(&Initializer::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

    // Wait until both threads have finished
    threadH.join();
    threadF.join();
    
    //FindHomography(vbMatchesInliersH, SH, H);
    //FindFundamental(vbMatchesInliersF, SF, F);
    
    // Compute ratio of scores
    float RH = SH/(SH+SF);
    
    //cout<< "SH: "<<SH<<endl;
    //cout<< "SF: "<<SF<<endl;
    cout<< "RH: "<<RH<<endl;
    
    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    if(RH>0.4)
        return ReconstructH(vbMatchesInliersH,H,mK,R21,t21,vP3D,vbTriangulated,1.0,50);
    else //if(RH<0.4)
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    return false;
}


void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    const int N = mvMatches12.size();

    // Normalize coordinates
    //vector<cv::Point2f> vPn1, vPn2;
    vector<cv::Point3f> vPn1, vPn2;
    cv::Mat T1, T2;
    //Normalize(mvKeys1,vPn1, T1);
    //Normalize(mvKeys2,vPn2, T2);
    Normalize3D(mvP3M1, vPn1, T1);
    Normalize3D(mvP3M2, vPn2, T2);
    
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    //vector<cv::Point2f> vPn1i(8);
    //vector<cv::Point2f> vPn2i(8);
    vector<cv::Point3f> vPn1i(8);
    vector<cv::Point3f> vPn2i(8);

    //vector<cv::Point3f> vPm1i(8);
    //vector<cv::Point3f> vPm2i(8);
    
    cv::Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;
    
    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int idx = mvSets[it][j];
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];

            //vPm1i[j] = mvP3M1[mvMatches12[idx].first];
            //vPm2i[j] = mvP3M2[mvMatches12[idx].second];
        }
        //cv::Mat Hn = ComputeH21(vPn1i,vPn2i);
        cv::Mat Hn = Compute3DH21(vPn1i,vPn2i);
        H21i = T2inv*Hn*T1;
        H12i = H21i.inv();
        //ComputeLambdaH(H21i, H12i, vPm1i, vPm2i);
        currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);
        
        if(currentScore>score)
        {
            H21 = H21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}


void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = vbMatchesInliers.size();

    // Normalize coordinates
    //vector<cv::Point2f> vPn1, vPn2;
    vector<cv::Point3f> vPn1, vPn2;
    cv::Mat T1, T2;
    //Normalize(mvKeys1,vPn1, T1);
    //Normalize(mvKeys2,vPn2, T2);
    Normalize3D(mvP3M1, vPn1, T1);
    Normalize3D(mvP3M2, vPn2, T2);
    
    cv::Mat T2t = T2.t();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    //vector<cv::Point2f> vPn1i(8);
    //vector<cv::Point2f> vPn2i(8);
    vector<cv::Point3f> vPn1i(8);
    vector<cv::Point3f> vPn2i(8);
    
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }
        
        //cv::Mat Fn = ComputeF21(vPn1i,vPn2i);
        cv::Mat Fn = Compute3DF21(vPn1i,vPn2i);

        F21i = T2t*Fn*T1;

        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}


cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(2*N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);
}

cv::Mat Initializer::Compute3DH21(const vector<cv::Point3f> &vP1, const vector<cv::Point3f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(2*N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float w1 = vP1[i].z;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;
        const float w2 = vP2[i].z;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -w2*u1;
        A.at<float>(2*i,4) = -w2*v1;
        A.at<float>(2*i,5) = -w2*w1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2*w1;

        A.at<float>(2*i+1,0) = w2*u1;
        A.at<float>(2*i+1,1) = w2*v1;
        A.at<float>(2*i+1,2) = w2;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2*w1;

    }
    
    cv::Mat u,w,vt;
    
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);
}

cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;
}

cv::Mat Initializer::Compute3DF21(const vector<cv::Point3f> &vP1, const vector<cv::Point3f> &vP2)
{
    const int N = vP1.size();

    cv::Mat A(N,9,CV_32F);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float w1 = vP1[i].z;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;
        const float w2 = vP2[i].z;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2*w1;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2*w1;
        A.at<float>(i,6) = w2*u1;
        A.at<float>(i,7) = w2*v1;
        A.at<float>(i,8) = w2*w1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;
}

float Initializer::CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma)
{   
    const int N = mvMatches12.size();

    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 5.991;

    //const float invSigmaSquare = 1.0/(sigma*sigma);

    const float invSigmaSquare = sigma*sigma;      //using EUCM model change the dimension ( 1/pixel -> 1/fx ), sigma = fx

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        //const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        //const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const cv::Point3f &p1 = mvP3M1[mvMatches12[i].first];
        const cv::Point3f &p2 = mvP3M2[mvMatches12[i].second];

        const float u1 = p1.x;
        const float v1 = p1.y;
        const float w1 = p1.z;
        const float u2 = p2.x;
        const float v2 = p2.y;
        const float w2 = p2.z;

        // Reprojection error in first image
        // x2in1 = H12*x2

        //const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        //const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        //const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;
        float u2in1 = h11inv*u2+h12inv*v2+h13inv*w2;
        float v2in1 = h21inv*u2+h22inv*v2+h23inv*w2;
        float w2in1 = h31inv*u2+h32inv*v2+h33inv*w2;
        
        cv::Mat x1a(3,1,CV_32F);
        cv::Mat x1b(3,1,CV_32F);
        x1a.at<float>(0,0) = u1;
        x1a.at<float>(1,0) = v1;
        x1a.at<float>(2,0) = w1;
        x1b.at<float>(0,0) = u2in1;
        x1b.at<float>(1,0) = v2in1;
        x1b.at<float>(2,0) = w2in1;
        
        //lambda*x1 = H'*x2, lambda*x1a = x1b, lambda is computed by Least squares method
        cv::Mat lambdaM1 = (x1a.t()*x1a).inv() * x1a.t() * x1b; 
        float lambdaS1 = lambdaM1.at<float>(0,0);
        //float lambdaS1 = mLambdaH12;
        //cout << "lambda1: " << lambdaS1<<endl;

        u2in1 = u2in1/lambdaS1;
        v2in1 = v2in1/lambdaS1;
        w2in1 = w2in1/lambdaS1;
        
        //const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);
        //const float squareDist1 = (lambdaS1*u1-u2in1)*(lambdaS1*u1-u2in1)+(lambdaS1*v1-v2in1)*(lambdaS1*v1-v2in1)+(lambdaS1*w1-w2in1)*(lambdaS1*w1-w2in1);
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1)+(w1-w2in1)*(w1-w2in1);
        //const float squareDist1 = (lambdaS1*u1-u2in1)*(lambdaS1*u1-u2in1)+(lambdaS1*v1-v2in1)*(lambdaS1*v1-v2in1);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        //cout << "HchiSquare1: " << chiSquare1<<endl;
        
        if(chiSquare1>th || lambdaS1 <= 0)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1

        //const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        //const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        //const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;
        float u1in2 = h11*u1+h12*v1+h13*w1;
        float v1in2 = h21*u1+h22*v1+h23*w1;
        float w1in2 = h31*u1+h32*v1+h33*w1;
        
        cv::Mat x2a(3,1,CV_32F);
        cv::Mat x2b(3,1,CV_32F);
        x2a.at<float>(0,0) = u2;
        x2a.at<float>(1,0) = v2;
        x2a.at<float>(2,0) = w2;
        x2b.at<float>(0,0) = u1in2;
        x2b.at<float>(1,0) = v1in2;
        x2b.at<float>(2,0) = w1in2;
        
        //lambda*x2 = H*x1, lambda*x2a = x2b, lambda is computed by Least squares method
        cv::Mat lambdaM2 = (x2a.t()*x2a).inv() * x2a.t() * x2b; 
        float lambdaS2 = lambdaM2.at<float>(0,0);
        //float lambdaS2 = mLambdaH21;
        //cout << "lambda2: " << lambdaS2<<endl;

        u1in2 = u1in2/lambdaS2;
        v1in2 = v1in2/lambdaS2;
        w1in2 = w1in2/lambdaS2;

        //const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);
        //const float squareDist2 = (lambdaS2*u2-u1in2)*(lambdaS2*u2-u1in2)+(lambdaS2*v2-v1in2)*(lambdaS2*v2-v1in2)+(lambdaS2*w2-w1in2)*(lambdaS2*w2-w1in2);
        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2)+(w2-w1in2)*(w2-w1in2);
        //const float squareDist2 = (lambdaS2*u2-u1in2)*(lambdaS2*u2-u1in2)+(lambdaS2*v2-v1in2)*(lambdaS2*v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        //cout << "HchiSquare2: " << chiSquare2<<endl;
        
        if(chiSquare2>th || lambdaS2 <= 0)
            bIn = false;
        else
            score += th - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    //const float invSigmaSquare = 1.0/(sigma*sigma);

    const float invSigmaSquare = sigma*sigma;       //using EUCM model change the dimension ( 1/pixel-> 1/fx )   sigma = fx

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        //const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        //const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const cv::Point3f &p1 = mvP3M1[mvMatches12[i].first];
        const cv::Point3f &p2 = mvP3M2[mvMatches12[i].second];

        const float u1 = p1.x;
        const float v1 = p1.y;
        const float w1 = p1.z;
        const float u2 = p2.x;
        const float v2 = p2.y;
        const float w2 = p2.z;
        
        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)

        //const float a2 = f11*u1+f12*v1+f13;
        //const float b2 = f21*u1+f22*v1+f23;
        //const float c2 = f31*u1+f32*v1+f33;
        const float a2 = f11*u1+f12*v1+f13*w1;
        const float b2 = f21*u1+f22*v1+f23*w1;
        const float c2 = f31*u1+f32*v1+f33*w1;

        //const float num2 = a2*u2+b2*v2+c2;
        const float num2 = a2*u2+b2*v2+c2*w2;

        //const float squareDist1 = num2*num2/(a2*a2+b2*b2);
        const float squareDist1 = num2*num2/(a2*a2+b2*b2+c2*c2);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        //cout << "FchiSquare1: " << chiSquare1<<endl;
        
        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        //const float a1 = f11*u2+f21*v2+f31;
        //const float b1 = f12*u2+f22*v2+f32;
        //const float c1 = f13*u2+f23*v2+f33;
        const float a1 = f11*u2+f21*v2+f31*w2;
        const float b1 = f12*u2+f22*v2+f32*w2;
        const float c1 = f13*u2+f23*v2+f33*w2;

        //const float num1 = a1*u1+b1*v1+c1;
        const float num1 = a1*u1+b1*v1+c1*w1;

        //const float squareDist2 = num1*num1/(a1*a1+b1*b1);
        const float squareDist2 = num1*num1/(a1*a1+b1*b1+c1*c1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        //cout << "FchiSquare2: " << chiSquare2<<endl;
        
        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix
    //cv::Mat E21 = K.t()*F21*K;
    cv::Mat E21 = F21;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    DecomposeE(E21,R1,R2,t);  
    
    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Reconstruct with the 4 hyphoteses and check
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;
    float parallax1,parallax2, parallax3, parallax4;

    /*int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);*/
    
    int nGood1 = CheckRT3D(R1,t1,mvP3M1,mvP3M2,mvMatches12,vbMatchesInliers, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT3D(R2,t1,mvP3M1,mvP3M2,mvMatches12,vbMatchesInliers, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT3D(R1,t2,mvP3M1,mvP3M2,mvMatches12,vbMatchesInliers, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT3D(R2,t2,mvP3M1,mvP3M2,mvMatches12,vbMatchesInliers, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);
    
    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1)
    {
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    //cv::Mat invK = K.inv();
    //cv::Mat A = invK*H21*K;
    cv::Mat A = H21;

    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();

    float s = cv::determinant(U)*cv::determinant(Vt);

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }


    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for(size_t i=0; i<8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;
        //int nGood = CheckRT(vR[i],vt[i],mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);
        int nGood = CheckRT3D(vR[i],vt[i],mvP3M1,mvP3M2,mvMatches12,vbMatchesInliers,vP3Di, 4.0*mSigma2, vbTriangulatedi, parallaxi);

        if(nGood>bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            secondBestGood = nGood;
        }
    }


    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax && bestGood>minTriangulated && bestGood>0.9*N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;

        return true;
    }

    return false;
}

void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

void Initializer::TriangulateFisheye(const cv::Point3f &p1m, const cv::Point3f &p2m, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = p1m.x*P1.row(2)-p1m.z*P1.row(0);
    A.row(1) = p1m.y*P1.row(2)-p1m.z*P1.row(1);
    A.row(2) = p2m.x*P2.row(2)-p2m.z*P2.row(0);
    A.row(3) = p2m.y*P2.row(2)-p2m.z*P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    float meanDevX = 0;
    float meanDevY = 0;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

void Initializer::Normalize3D(const vector<cv::Point3f> &vP3M, vector<cv::Point3f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = vP3M.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vP3M[i].x;
        meanY += vP3M[i].y;
    }

    meanX = meanX/N;
    meanY = meanY/N;
    
    float meanDevX = 0;
    float meanDevY = 0;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vP3M[i].x - meanX*vP3M[i].z;
        vNormalizedPoints[i].y = vP3M[i].y - meanY*vP3M[i].z;
        vNormalizedPoints[i].z = vP3M[i].z;
        
        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }
    
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;
    
    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }
    
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(vKeys1.size(),false);
    vP3D.resize(vKeys1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    cv::Mat O2 = -R.t()*t;

    int nGood=0;

    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        if(!vbMatchesInliers[i])
            continue;

        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;

        Triangulate(kp1,kp2,P1,P2,p3dC1);

        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;

        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image
        float im1x, im1y;
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        if(squareError2>th2)
            continue;

        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        nGood++;

        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}

int Initializer::CheckRT3D(const cv::Mat &R, const cv::Mat &t, const vector<cv::Point3f> &vP3M1, const vector<cv::Point3f> &vP3M2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    
    vbGood = vector<bool>(vP3M1.size(),false);
    vP3D.resize(vP3M1.size());

    vector<float> vCosParallax;
    vCosParallax.reserve(vP3M1.size());

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    cv::Mat O2 = -R.t()*t;

    int nGood=0;
    
    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        if(!vbMatchesInliers[i])
            continue;
        
        const cv::Point3f &p1m = vP3M1[vMatches12[i].first];
        const cv::Point3f &p2m = vP3M2[vMatches12[i].second];
        cv::Mat p3dC1;
        
        TriangulateFisheye(p1m, p2m, P1, P2, p3dC1);
        
        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }
        
        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2)/(dist1*dist2);
        
        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;
        
        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;

        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;
        
        // Check reprojection error in first image
        float im1x, im1y;
        float im1z;
        float p3dC1x = p3dC1.at<float>(0);
        float p3dC1y = p3dC1.at<float>(1);
        float p3dC1z = p3dC1.at<float>(2);
        float im1d = sqrt( mfBeta*(p3dC1x*p3dC1x+p3dC1y*p3dC1y) + p3dC1z*p3dC1z );
        
        im1x = p3dC1x / ( mfAlpha*im1d + (1-mfAlpha)*p3dC1z );
        im1y = p3dC1y / ( mfAlpha*im1d + (1-mfAlpha)*p3dC1z );
        
        float distance1R2 = im1x*im1x + im1y*im1y;
        //EUCM model unprojection range, R2 = Mx^2+My^2
        if(distance1R2>mR2range)
        {
            cout<<"out of the range"<<endl;
            continue;
        }

        im1z = ( 1-mfBeta*mfAlpha*mfAlpha*distance1R2 )/( mfAlpha*sqrt( 1-(2*mfAlpha-1)*mfBeta*distance1R2 ) + 1-mfAlpha );
        
        float squareError1 = (im1x-p1m.x)*(im1x-p1m.x)+(im1y-p1m.y)*(im1y-p1m.y)+(im1z-p1m.z)*(im1z-p1m.z);
        
        //cout << "Trangulate squareError1: " << squareError1<<endl;
        
        if(squareError1>th2)
            continue;
        
        // Check reprojection error in second image
        float im2x, im2y;
        float im2z;
        float p3dC2x = p3dC2.at<float>(0);
        float p3dC2y = p3dC2.at<float>(1);
        float p3dC2z = p3dC2.at<float>(2);
        float im2d = sqrt( mfBeta*(p3dC2x*p3dC2x+p3dC2y*p3dC2y) + p3dC2z*p3dC2z );
        
        im2x = p3dC2x / ( mfAlpha*im2d + (1-mfAlpha)*p3dC2z );
        im2y = p3dC2y / ( mfAlpha*im2d + (1-mfAlpha)*p3dC2z );
        float distance2R2 = im2x*im2x + im2y*im2y;

        //EUCM model unprojection range, R2 = Mx^2+My^2
        if(distance2R2>mR2range)
        {
            cout<<"out of the range"<<endl;
            continue;
        }

        im2z = ( 1-mfBeta*mfAlpha*mfAlpha*distance2R2 )/( mfAlpha*sqrt( 1-(2*mfAlpha-1)*mfBeta*distance2R2 ) + 1-mfAlpha );
        
        float squareError2 = (im2x-p2m.x)*(im2x-p2m.x)+(im2y-p2m.y)*(im2y-p2m.y)+(im2z-p2m.z)*(im2z-p2m.z);
        
        //cout << "Trangulate squareError2: " << squareError2<<endl;
        
        if(squareError2>th2)
            continue;
        
        vCosParallax.push_back(cosParallax);
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        nGood++;

        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }
    
    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;
    
    return nGood;
}

void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

void Initializer::ComputeLambdaH(const cv::Mat &H21, const cv::Mat &H12, const vector<cv::Point3f> &vP1, const vector<cv::Point3f> &vP2)
{
    mLambdaH12 = 0;
    mLambdaH21 = 0;

    const int N = vP1.size();

    cv::Mat A1(3,1,CV_32F);
    cv::Mat B1(3,1,CV_32F);
    cv::Mat A2(3,1,CV_32F);
    cv::Mat B2(3,1,CV_32F);

    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float w1 = vP1[i].z;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;
        const float w2 = vP2[i].z;

        const float u2in1 = h11inv*u2+h12inv*v2+h13inv*w2;
        const float v2in1 = h21inv*u2+h22inv*v2+h23inv*w2;
        const float w2in1 = h31inv*u2+h32inv*v2+h33inv*w2;

        A1.at<float>(0,0) = u1;
        A1.at<float>(1,0) = v1;
        A1.at<float>(2,0) = w1;
        B1.at<float>(0,0) = u2in1;
        B1.at<float>(1,0) = v2in1;
        B1.at<float>(2,0) = w2in1;

        cv::Mat lambdaM1 = (A1.t()*A1).inv() * A1.t() *B1;     // A1*lambdaM1 = B1
        mLambdaH12 += lambdaM1.at<float>(0,0);

        const float u1in2 = h11*u1+h12*v1+h13*w1;
        const float v1in2 = h21*u1+h22*v1+h23*w1;
        const float w1in2 = h31*u1+h32*v1+h33*w1;

        A2.at<float>(0,0) = u2;
        A2.at<float>(1,0) = v2;
        A2.at<float>(2,0) = w2;
        B2.at<float>(0,0) = u1in2;
        B2.at<float>(1,0) = v1in2;
        B2.at<float>(2,0) = w1in2;

        cv::Mat lambdaM2 = (A2.t()*A2).inv() * A2.t() *B2;    // A2*lambdaM2 = B2
        mLambdaH21 += lambdaM2.at<float>(0,0);
    }

    mLambdaH12 = mLambdaH12/N;
    mLambdaH21 = mLambdaH21/N;

    //cout<<"mLambdaH12: "<<mLambdaH12<<endl;
    //cout<<"mLambdaH21: "<<mLambdaH21<<endl;

}

} //namespace ORB_SLAM
