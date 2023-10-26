
#include "margin.h"

void ResidualBlockInfo::Evaluate()
{
    residuals.resize(cost_function->num_residuals());

    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    for(int i = 0; i < static_cast<int>(block_sizes.size()); ++i)
    {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
    }
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    if(loss_function)
    {
        double sq_norm, rho[3];
        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);

        double sqrt_rho1_ = sqrt(rho[1]);
        for(int i = 0; i < static_cast<int>(parameter_blocks.size()); ++i)
        {
            jacobians[i] *= sqrt_rho1_;
        }
        residuals *= sqrt_rho1_;
    }
}



void MarginalizationInfo::AddResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    factors.emplace_back(residual_block_info);

    std::vector<double*> &parameter_blocks = residual_block_info->parameter_blocks;
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    for(int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); ++i)
    {
        double *addr = parameter_blocks[i];
        int size = parameter_block_sizes[i];
        parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }
    for(int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); ++i)
    {
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}


void MarginalizationInfo::PreMarginalize()
{
    for(auto it : factors)
    {
        it->Evaluate();
        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for(int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            int size = block_sizes[i];
            if(parameter_block_data.find(addr) == parameter_block_data.end())
            {
                double *data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;
            }
        }
    }
}


MarginalizationInfo::~MarginalizationInfo()
{
    for(auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete[] it->second;
    for (int i = 0; i < (int)factors.size(); i++)
    {
        delete[] factors[i]->raw_jacobians;
        delete factors[i]->cost_function;
        delete factors[i];
    }
}

inline int LocalSize(int size)
{
    if(integration_mode == IntegrationMode::SE23)
        return size;
    else if(integration_mode == IntegrationMode::SO3)
        return size == 9 ? 3 : size;
    else if(integration_mode == IntegrationMode::Quaternion)
        return size == 4 ? 3 : size;
    else
        return size;
}

void* LinearConstruct(void* threadsstruct)
{
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
    for(auto it : p->sub_factors)
    {
        for(int i = 0; i < static_cast<int>(it->parameter_blocks.size()); ++i)
        {
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];

            size_i = LocalSize(size_i);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for(int j = i; j < static_cast<int>(it->parameter_blocks.size()); ++j)
            {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                size_j = LocalSize(size_j);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose()*it->residuals;
        }
    }
    return threadsstruct;
}

void MarginalizationInfo::Marginalize()
{

    int pos = 0;
    for(auto &it : parameter_block_idx)
    {
        it.second = pos;
        pos += LocalSize(parameter_block_size[it.first]);
    }

    m = pos;

    for(const auto &it : parameter_block_size)
    {
        if(parameter_block_idx.find(it.first) == parameter_block_idx.end())
        {
            parameter_block_idx[it.first] = pos;
            pos += LocalSize(it.second);
        }
    }

    n = pos - m;

    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();

    pthread_t tids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];
    int i = 0;
    for(auto it : factors)
    {
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS;
    }
    for(int i = 0; i < NUM_THREADS; ++i)
    {
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size;
        threadsstruct[i].parameter_block_idx = parameter_block_idx;
        int ret = pthread_create(&tids[i], nullptr, LinearConstruct ,(void*)&(threadsstruct[i]));
        if (ret != 0)
        {
            std::cout << "pthread_create error !!!" << std::endl;
        }
    }
    for( int i = NUM_THREADS - 1; i >= 0; --i)
    {
        pthread_join(tids[i], NULL);
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }

    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();

    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    prior_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();

    prior_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;

}

std::vector<double*> MarginalizationInfo::GetParameterBlocks(std::unordered_map<long, double*> &addr_shift)
{
    std::vector<double*> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for(const auto &it : parameter_block_idx)
    {
        if(it.second >= m)
        {
            keep_block_size.push_back(parameter_block_size[it.first]);
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            keep_block_data.push_back(parameter_block_data[it.first]);
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}


MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _margin):marginalization_info(_margin)
{
    for(auto it : marginalization_info->keep_block_size)
        mutable_parameter_block_sizes()->push_back(it);

    set_num_residuals(marginalization_info->n);
};

bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    int m = marginalization_info->m;
    int n = marginalization_info->n;
    Eigen::VectorXd dx(n);
    for(int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); ++i)
    {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;
        if(size == 9)
        {
            if(integration_mode == IntegrationMode::SE23)
            {
                Eigen::VectorXd x = Eigen::Map<const Eigen::Vector9d>(parameters[i]);
                Eigen::VectorXd x0 = Eigen::Map<const Eigen::Vector9d>(marginalization_info->keep_block_data[i]);
                dx.segment<9>(idx) = LogSE23(InverseSE23(ExpSE23(x0))*ExpSE23(x));
            }
            else if(integration_mode == IntegrationMode::SO3)
            {
                Eigen::Matrix3d x = Eigen::Map<const Eigen::Matrix3d>(parameters[i]);
                Eigen::Matrix3d x0 = Eigen::Map<const Eigen::Matrix3d>(marginalization_info->keep_block_data[i]);
                dx.segment<3>(idx) = LogSO3(x0.transpose()*x);
            }
        }
        else if(size == 6)
        {
            Eigen::VectorXd x = Eigen::Map<const Eigen::Vector6d>(parameters[i]);
            Eigen::VectorXd x0 = Eigen::Map<const Eigen::Vector6d>(marginalization_info->keep_block_data[i]);
            dx.segment<6>(idx) = x - x0;
        }
        else if(size == 4)
        {
            Eigen::VectorXd x = Eigen::Map<const Eigen::Vector4d>(parameters[i]);
            Eigen::VectorXd x0 = Eigen::Map<const Eigen::Vector4d>(marginalization_info->keep_block_data[i]);
            dx.segment<3>(idx) = 2.0*(Eigen::Quaterniond(x0(3), x0(0), x0(1), x0(2)).inverse()*Eigen::Quaterniond(x(3), x(0), x(1), x(2))).vec();
            if(!((Eigen::Quaterniond(x0(3), x0(0), x0(1), x0(2)).inverse() * Eigen::Quaterniond(x(3), x(0), x(1), x(2))).w() >= 0))
            {
                dx.segment<3>(idx) = -2.0*(Eigen::Quaterniond(x0(3), x0(0), x0(1), x0(2)).inverse() * Eigen::Quaterniond(x(3), x(0), x(1), x(2))).vec();
            }
        }
        else if(size == 3)
        {
            Eigen::VectorXd x = Eigen::Map<const Eigen::Vector3d>(parameters[i]);
            Eigen::VectorXd x0 = Eigen::Map<const Eigen::Vector3d>(marginalization_info->keep_block_data[i]);
            dx.segment<3>(idx) = x - x0;
        }
    }
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->prior_residuals + marginalization_info->prior_jacobians*dx;
    if(jacobians)
    {
        for(int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); ++i)
        {
            if(jacobians[i])
            {
                int size = marginalization_info->keep_block_size[i];
                int local_size = LocalSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobians[i], n, size);
                J.setZero();
                J.leftCols(local_size) = marginalization_info->prior_jacobians.middleCols(idx, local_size);
            }
        }
    }
    return true;
}
